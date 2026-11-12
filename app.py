
import os, tempfile, zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

st.set_page_config(page_title="Football Motion Analyzer", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.main {background:#0b1220;}
.block-container {padding-top:2rem; max-width:1400px;}
h1,h2,h3 {letter-spacing:-.03em;}
div[data-testid="stMetric"] {background:#111c2e;border:1px solid #26364e;padding:16px;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.title("⚽ Football Motion Analyzer")
st.caption("2D single-camera tracking · 3D dual-camera reconstruction")
st.info("First version is configured for iPhone 13 (Camera A) + Motorola Edge 50 Neo (Camera B), both using the 1× rear camera. Keep both phones stationary and record the same action at matching frame rates.")

mode = st.radio("Choose your setup", ["Single camera · 2D", "Two cameras · 3D"], horizontal=True)

def save_upload(upload):
    suffix = Path(upload.name).suffix or ".mp4"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(upload.getbuffer()); f.close()
    return f.name

def read_video_info(path):
    cap = cv2.VideoCapture(path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if info["fps"] <= 0 or info["frames"] <= 0:
        raise ValueError("Could not read this video. Try MP4 (H.264) footage.")
    info["duration"] = info["frames"] / info["fps"]
    return info

def fig_to_placeholder():
    return

def single_camera_analysis(path, min_area, max_area, min_ratio, max_ratio, xlo, xhi, ylo, yhi):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: raise ValueError("Invalid video frame rate.")
    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    kernel = np.ones((3,3), np.uint8)
    records=[]; annotated=[]; frame_id=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        mask=bg.apply(frame)
        mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for c in contours:
            area=cv2.contourArea(c)
            if min_area < area < max_area:
                x,y,w,h=cv2.boundingRect(c); ratio=w/max(h,1)
                if min_ratio < ratio < max_ratio:
                    candidates.append((area,x,y,w,h))
        x=y=None; box=None
        if candidates:
            area,bx,by,w,h=max(candidates,key=lambda z:z[0])
            x=bx+w/2; y=by+h/2; box=(bx,by,bx+w,by+h)
        records.append({"frame":frame_id,"time":frame_id/fps,"x":x,"y":y})
        if box:
            cv2.rectangle(frame,box[:2],box[2:],(0,255,0),2)
            cv2.circle(frame,(int(x),int(y)),5,(0,0,255),-1)
        cv2.putText(frame,"SINGLE CAMERA - BALL CANDIDATE",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
        annotated.append(frame)
        frame_id+=1
    cap.release()
    df=pd.DataFrame(records)
    clean=df.dropna(subset=["x","y"]).copy()
    h,w=read_video_info(path)["height"],read_video_info(path)["width"]
    clean=clean[(clean.x>xlo*w)&(clean.x<xhi*w)&(clean.y>ylo*h)&(clean.y<yhi*h)].copy()
    if len(clean)<5: raise ValueError(f"Only {len(clean)} usable detections. Adjust the ROI and contour settings, improve lighting, or use a clearer clip.")
    t=clean.time.to_numpy(); x=clean.x.to_numpy(); y=clean.y.to_numpy()
    xcoef=np.polyfit(t,x,1); ycoef=np.polyfit(t,y,2)
    a,b,c=ycoef
    peak_t=-b/(2*a) if abs(a)>1e-12 else np.nan
    peak_x=float(np.polyval(xcoef,peak_t)) if np.isfinite(peak_t) else np.nan
    peak_y=float(np.polyval(ycoef,peak_t)) if np.isfinite(peak_t) else np.nan
    fit=pd.DataFrame({"time":np.linspace(t.min(),t.max(),300)})
    fit["x_fit"]=np.polyval(xcoef,fit.time); fit["y_fit"]=np.polyval(ycoef,fit.time)
    # Save annotated video
    out_path=tempfile.mktemp(suffix=".mp4")
    H,W=annotated[0].shape[:2]
    writer=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
    for fr in annotated: writer.write(fr)
    writer.release()
    trail_path = write_red_trail_video(path, clean, "x", "y", "frame", fps)
    return df,clean,fit,xcoef,ycoef,peak_t,peak_x,peak_y,out_path,trail_path


def write_red_trail_video(source_path, track_df, x_col="x", y_col="y", frame_col="frame", fps_override=None):
    """Replay the full source clip with a persistent red polyline through tracked ball centres."""
    cap = cv2.VideoCapture(source_path)
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    points_by_frame = {}
    for _, row in track_df.iterrows():
        if pd.notna(row.get(x_col)) and pd.notna(row.get(y_col)):
            points_by_frame[int(row[frame_col])] = (int(round(row[x_col])), int(round(row[y_col])))
    trail = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in points_by_frame:
            trail.append(points_by_frame[frame_idx])
        # Red trajectory line, with a red marker at the latest tracked ball centre.
        for p1, p2 in zip(trail[:-1], trail[1:]):
            cv2.line(frame, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)
        if trail:
            cv2.circle(frame, trail[-1], 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "BALL TRAJECTORY", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return out_path

def detect_yolo(path, camera_label, model, conf, imgsz, max_frames):
    cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS)
    records=[]; frames=[]; fid=0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride=max(1,int(np.ceil(total/max_frames))) if max_frames else 1
    progress=st.progress(0, text=f"Detecting ball in Camera {camera_label}…")
    while True:
        ok,frame=cap.read()
        if not ok: break
        if fid % stride == 0:
            result=model.predict(frame,imgsz=imgsz,conf=conf,classes=[32],verbose=False)[0]
            detected=False; x=y=x1=y1=x2=y2=score=np.nan
            if result.boxes is not None and len(result.boxes)>0:
                boxes=result.boxes.xyxy.cpu().numpy(); scores=result.boxes.conf.cpu().numpy()
                k=int(np.argmax(scores)); x1,y1,x2,y2=boxes[k]
                x=(x1+x2)/2; y=(y1+y2)/2; score=float(scores[k]); detected=True
                vis=result.plot()
                frames.append((fid,vis))
            records.append({"camera":camera_label,"frame":fid,"time":fid/fps,"detected":detected,
                            "x":x,"y":y,"confidence":score,"x1":x1,"y1":y1,"x2":x2,"y2":y2})
        fid+=1
        if fid%max(1,int(fps))==0: progress.progress(min(.99,fid/max(total,1)),text=f"Camera {camera_label}: {fid}/{total} frames")
    cap.release(); progress.empty()
    df=pd.DataFrame(records)
    # Match notebook behaviour: interpolate only short internal gaps (<=3 sampled frames).
    data=df.sort_values("frame").reset_index(drop=True).copy()
    data["x_clean"]=data.x; data["y_clean"]=data.y
    valid=np.where(data.x.notna()&data.y.notna())[0]
    for left,right in zip(valid[:-1],valid[1:]):
        if 0 < right-left-1 <= 3:
            for j in range(left+1,right):
                alpha=(j-left)/(right-left)
                data.loc[j,"x_clean"]=data.loc[left,"x"]*(1-alpha)+data.loc[right,"x"]*alpha
                data.loc[j,"y_clean"]=data.loc[left,"y"]*(1-alpha)+data.loc[right,"y"]*alpha
    data["interpolated"]=data.x.isna()&data.x_clean.notna()
    return data,frames,stride



def write_sampled_detection_video(frames, source_path, stride):
    """Write annotated YOLO frames as a review video at the sampled-frame rate."""
    if not frames:
        raise ValueError("No annotated frames were generated.")
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    out_fps = max(1.0, fps / max(1, stride))
    h, w = frames[0][1].shape[:2]
    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    for _, frame in frames:
        writer.write(frame)
    writer.release()
    return out_path

def build_stereo_video(pathA,pathB,trackA,trackB,physical=None):
    capA=cv2.VideoCapture(pathA); capB=cv2.VideoCapture(pathB)
    fps=min(capA.get(cv2.CAP_PROP_FPS),capB.get(cv2.CAP_PROP_FPS))
    start=max(trackA.time.min(),trackB.time.min()); end=min(trackA.time.max(),trackB.time.max())
    if end<=start: raise ValueError("The two clips do not have an overlapping time interval.")
    capA.set(cv2.CAP_PROP_POS_MSEC,start*1000); capB.set(cv2.CAP_PROP_POS_MSEC,start*1000)
    okA,fa=capA.read(); okB,fb=capB.read()
    if not okA or not okB: raise ValueError("Could not read synchronized video frames.")
    th=max(fa.shape[0],fb.shape[0]); wa=int(fa.shape[1]*th/fa.shape[0]); wb=int(fb.shape[1]*th/fb.shape[0])
    panelw=390; outpath=tempfile.mktemp(suffix=".mp4")
    writer=cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc(*"mp4v"),fps,(wa+wb+panelw,th))
    capA.set(cv2.CAP_PROP_POS_MSEC,start*1000); capB.set(cv2.CAP_PROP_POS_MSEC,start*1000)
    trails=[[],[]]
    def nearest(df,t):
        idx=(df.time-t).abs().idxmin(); row=df.loc[idx]
        return row if abs(row.time-t)<=max(.04,1/fps) else None
    n=0
    while start+n/fps<=end:
        ts=start+n/fps; okA,fa=capA.read(); okB,fb=capB.read()
        if not okA or not okB: break
        for fr,df,tr,label in [(fa,trackA,trails[0],"IPHONE 13"),(fb,trackB,trails[1],"MOTOROLA EDGE 50 NEO")]:
            row=nearest(df,ts)
            if row is not None and pd.notna(row.x_clean):
                px,py=int(row.x_clean),int(row.y_clean); tr.append((px,py))
                cv2.circle(fr,(px,py),7,(0,0,255),-1)
                for p,q in zip(tr[:-1],tr[1:]): cv2.line(fr,p,q,(0,0,255),3)
                if pd.notna(row.x1): cv2.rectangle(fr,(int(row.x1),int(row.y1)),(int(row.x2),int(row.y2)),(0,255,0),2)
            cv2.putText(fr,label,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
        fa=cv2.resize(fa,(wa,th)); fb=cv2.resize(fb,(wb,th))
        panel=np.zeros((th,panelw,3),np.uint8); panel[:]=(28,32,42)
        def put(s,y,scale=.65,col=(255,255,255)): cv2.putText(panel,s,(18,y),cv2.FONT_HERSHEY_SIMPLEX,scale,col,2,cv2.LINE_AA)
        put("FOOTBALL MOTION",48,.8,(0,220,255)); put(f"Video time: {ts:.3f}s",95)
        if physical is not None:
            elapsed=ts-float(physical.time_elapsed.iloc[0]+start if False else physical.attrs.get("release_time",start))
            if physical.time_elapsed.min()<=elapsed<=physical.time_elapsed.max():
                xm=np.interp(elapsed,physical.time_elapsed,physical.X_m); ym=np.interp(elapsed,physical.time_elapsed,physical.Y_m)
                put(f"Elapsed: {elapsed:.3f}s",165,.7,(0,255,255)); put(f"X: {xm:+.3f} m",235,.9,(100,255,100)); put(f"Y: {ym:+.3f} m",305,.9,(100,200,255))
            else: put("Position panel unavailable",180,.55)
        else: put("3D overlay not available",180,.55)
        put("Estimated values; not calibrated",th-35,.42)
        writer.write(np.hstack([fa,fb,panel])); n+=1
    capA.release(); capB.release(); writer.release()
    return outpath

def plot_line(x,y,title,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(x,y,"o-",markersize=3); ax.set(title=title,xlabel=xlabel,ylabel=ylabel); ax.grid(True,alpha=.3); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

if mode.startswith("Single"):
    st.subheader("Single-camera 2D analysis")
    st.write("Upload one clip. The detector uses the notebook's MOG2 background-subtraction and contour approach.")
    up=st.file_uploader("Upload video",type=["mp4","mov","avi","m4v"],key="single")
    with st.expander("Detection settings",expanded=False):
        c1,c2,c3,c4=st.columns(4)
        min_area=c1.number_input("Minimum contour area",1,100000,50)
        max_area=c2.number_input("Maximum contour area",10,500000,5000)
        min_ratio=c3.number_input("Minimum width/height",0.1,5.0,0.6,.1)
        max_ratio=c4.number_input("Maximum width/height",0.2,10.0,1.6,.1)
        st.caption("Region of interest (fraction of frame). Defaults cover the full frame; adjust if background motion creates false detections.")
        r1,r2,r3,r4=st.columns(4)
        xlo=r1.slider("Left",0.0,1.0,0.0,.01); xhi=r2.slider("Right",0.0,1.0,1.0,.01)
        ylo=r3.slider("Top",0.0,1.0,0.0,.01); yhi=r4.slider("Bottom",0.0,1.0,1.0,.01)
    if up:
        p=save_upload(up)
        try:
            info=read_video_info(p); st.caption(f"{info['width']}×{info['height']} · {info['fps']:.2f} FPS · {info['duration']:.2f} seconds")
        except Exception as e: st.error(str(e)); st.stop()
        if st.button("Analyze single-camera video",type="primary"):
            with st.spinner("Running 2D tracking and curve fitting…"):
                try:
                    result=single_camera_analysis(p,min_area,max_area,min_ratio,max_ratio,xlo,xhi,ylo,yhi)
                    st.session_state["single_result"]=result
                except Exception as e: st.error(str(e))
    if "single_result" in st.session_state:
        df,clean,fit,xc,yc,pt,px,py,vpath,trailpath=st.session_state["single_result"]
        st.success(f"Analysis complete · {len(clean)} trajectory points retained")
        a,b,c=st.columns(3); a.metric("Detected frames",f"{df.x.notna().sum()}"); b.metric("Points fitted",len(clean)); c.metric("Estimated peak time",f"{pt:.3f} s" if np.isfinite(pt) else "N/A")
        st.markdown("#### Fitted equations")
        st.code(f"x(t) = {xc[0]:.4f} t + {xc[1]:.4f}  [pixels]\ny(t) = {yc[0]:.4f} t² + {yc[1]:.4f} t + {yc[2]:.4f}  [pixels]")
        if np.isfinite(pt): st.write(f"Fitted vertical turning point: **({px:.1f}, {py:.1f}) pixels** at **{pt:.3f} s**. This is the parabola's mathematical vertex and may lie outside the observed interval.")
        c1,c2=st.columns(2)
        with c1: plot_line(clean.x,clean.y,"Detected 2D trajectory","X (pixels)","Y (pixels)"); st.pyplot(plt.gcf()) if False else None
        with c2:
            fig,ax=plt.subplots(figsize=(8,4)); ax.scatter(clean.time,clean.y,s=12,label="detections"); ax.plot(fit.time,fit.y_fit,label="quadratic fit"); ax.invert_yaxis(); ax.set(xlabel="Time (s)",ylabel="Y (pixels)",title="Vertical coordinate vs time"); ax.legend(); ax.grid(alpha=.3); st.pyplot(fig); plt.close(fig)
        st.markdown("#### Video 1 — Detection / bounding-box tracking")
        st.video(vpath)
        st.markdown("#### Video 2 — Red-line trajectory trail")
        st.video(trailpath)
        st.download_button("Download detection video",open(vpath,"rb").read(),"single_camera_detection_tracking.mp4","video/mp4")
        st.download_button("Download red-line trajectory video",open(trailpath,"rb").read(),"single_camera_red_trajectory.mp4","video/mp4")
        st.download_button("Download detections CSV",df.to_csv(index=False).encode(),"single_camera_detections.csv","text/csv")
        st.download_button("Download fitted trajectory CSV",clean.to_csv(index=False).encode(),"single_camera_trajectory.csv","text/csv")
        with open(vpath,"rb") as f: st.download_button("Download annotated video",f.read(),"single_camera_tracking.mp4","video/mp4")

else:
    st.subheader("Dual-camera 3D analysis")
    st.write("Upload synchronized footage from the iPhone 13 and Motorola Edge 50 Neo. Both should record the same action with minimal start-time offset.")
    colA,colB=st.columns(2)
    upA=colA.file_uploader("Camera A · iPhone 13 · 1×",type=["mp4","mov","avi","m4v"],key="camA")
    upB=colB.file_uploader("Camera B · Motorola Edge 50 Neo · 1×",type=["mp4","mov","avi","m4v"],key="camB")
    with st.expander("Processing settings",expanded=False):
        conf=st.slider("YOLO confidence threshold",0.01,.8,.10,.01)
        imgsz=st.select_slider("YOLO image size",options=[640,960,1280],value=1280)
        max_frames=st.select_slider("Maximum sampled frames per camera (0 = all frames)",options=[0,100,200,400,800],value=0)
        tolerance=st.slider("Maximum camera timestamp difference (ms)",5,100,20,1)
        ball_diameter=st.number_input("Assumed football diameter (metres)",.10,.30,.22,.01)
    if upA and upB:
        pa=save_upload(upA); pb=save_upload(upB)
        try:
            ia=read_video_info(pa); ib=read_video_info(pb)
            c1,c2=st.columns(2)
            c1.caption(f"iPhone: {ia['width']}×{ia['height']} · {ia['fps']:.2f} FPS · {ia['duration']:.2f}s")
            c2.caption(f"Motorola: {ib['width']}×{ib['height']} · {ib['fps']:.2f} FPS · {ib['duration']:.2f}s")
            st.markdown("#### Uploaded camera footage")
            vA, vB = st.columns(2)
            with vA:
                st.caption("Camera A — iPhone 13")
                st.video(pa)
            with vB:
                st.caption("Camera B — Motorola Edge 50 Neo")
                st.video(pb)
        except Exception as e: st.error(str(e)); st.stop()
        if abs(ia["fps"]-ib["fps"])>1.0: st.warning("The frame rates differ significantly. Timestamp matching may produce fewer stereo pairs.")
        if st.button("Analyze both cameras",type="primary"):
            try:
                from ultralytics import YOLO
                model=YOLO("yolov8s.pt")
                with st.spinner("Detecting the football in both videos with YOLOv8s…"):
                    ta,framesA,strideA=detect_yolo(pa,"A",model,conf,imgsz,max_frames)
                    tb,framesB,strideB=detect_yolo(pb,"B",model,conf,imgsz,max_frames)
                    annotatedA = write_sampled_detection_video(framesA, pa, strideA)
                    annotatedB = write_sampled_detection_video(framesB, pb, strideB)
                A=ta[ta.detected].copy().reset_index(drop=True); B=tb[tb.detected].copy().reset_index(drop=True)
                if min(len(A),len(B))<8: raise ValueError("Fewer than 8 ball detections in one camera. Try a clearer clip or lower the confidence threshold.")
                cost=np.abs(A.time.to_numpy()[:,None]-B.time.to_numpy()[None,:])
                rr,cc=linear_sum_assignment(cost); keep=cost[rr,cc] <= tolerance/1000
                A=A.iloc[rr[keep]].reset_index(drop=True); B=B.iloc[cc[keep]].reset_index(drop=True)
                pairs=pd.DataFrame({"time":(A.time+B.time)/2,"x_A":A.x,"y_A":A.y,"x_B":B.x,"y_B":B.y,
                                    "time_A":A.time,"time_B":B.time,"time_diff_ms":np.abs(A.time-B.time)*1000})
                redTrailA = write_red_trail_video(pa, ta, "x_clean", "y_clean", "frame", ia["fps"])
                redTrailB = write_red_trail_video(pb, tb, "x_clean", "y_clean", "frame", ib["fps"])
                if len(pairs)<8: raise ValueError(f"Only {len(pairs)} synchronized ball pairs found. Need at least 8; current tolerance is {tolerance} ms.")
                # Notebook's approximate phone intrinsics, scaled to each original video resolution.
                # Notebook estimates fx=661.9/611 at 960x540.
                def intrinsics(info, focal):
                    sx=info["width"]/960; sy=info["height"]/540
                    return np.array([[focal*sx,0,(info["width"]-1)/2],[0,focal*sy,(info["height"]-1)/2],[0,0,1]],dtype=np.float64)
                KA=intrinsics(ia,661.9); KB=intrinsics(ib,611.0)
                ptsA=pairs[["x_A","y_A"]].to_numpy(np.float64); ptsB=pairs[["x_B","y_B"]].to_numpy(np.float64)
                normA=cv2.undistortPoints(ptsA.reshape(-1,1,2),KA,None).reshape(-1,2)
                normB=cv2.undistortPoints(ptsB.reshape(-1,1,2),KB,None).reshape(-1,2)
                E,emask=cv2.findEssentialMat(normA,normB,np.eye(3),method=cv2.RANSAC,prob=.999,threshold=.001)
                if E is None: raise ValueError("Essential matrix estimation failed. Check synchronization and camera views.")
                _,R,T,pmask=cv2.recoverPose(E,normA,normB,np.eye(3))
                valid=pmask.ravel()>0
                p1=normA[valid]; p2=normB[valid]
                P1=np.hstack([np.eye(3),np.zeros((3,1))]); P2=np.hstack([R,T])
                p4=cv2.triangulatePoints(P1,P2,p1.T,p2.T)
                xyz=(p4[:3]/p4[3]).T
                cam2=(R@xyz.T+T).T
                depth=np.isfinite(xyz).all(axis=1)&(xyz[:,2]>0)&(cam2[:,2]>0)
                xyz=xyz[depth]; valid_indices=np.where(valid)[0][depth]
                times=pairs.time.to_numpy()[valid_indices]
                if len(xyz)<5: raise ValueError("Fewer than 5 valid positive-depth triangulated points. The camera geometry could not be recovered reliably.")
                order=np.argsort(times); times=times[order]; xyz=xyz[order]
                traj=pd.DataFrame({"time":times,"X":xyz[:,0],"Y":xyz[:,1],"Z":xyz[:,2]})
                fitdf=traj.iloc[:-1].copy() if len(traj)>6 else traj.copy()
                t=fitdf.time.to_numpy(); t0=t[0]; tau=t-t0; fits={}
                for axis in ["X","Y","Z"]: fits[axis]=np.polyfit(tau,fitdf[axis].to_numpy(),2)
                velocity={k:2*fits[k][0]*tau+fits[k][1] for k in fits}
                acceleration={k:np.full(len(t),2*fits[k][0]) for k in fits}
                speed=np.sqrt(sum(v*v for v in velocity.values()))
                accmag=float(np.sqrt(sum(a[0]**2 for a in acceleration.values())))
                motion=pd.DataFrame({"Time":t,**{f"V{k}":velocity[k] for k in fits},"Speed":speed,
                                     **{f"A{k}":acceleration[k] for k in fits},"Acceleration":np.sqrt(sum(a*a for a in acceleration.values()))})
                # Approximate ball-size scale estimate, as in the notebook.
                scales=[]
                for _,row in traj.iterrows():
                    ix=(ta.time-row.time).abs().idxmin(); det=ta.loc[ix]
                    if abs(det.time-row.time)>max(.04,1/ia["fps"]): continue
                    bw=float(det.x2-det.x1) if pd.notna(det.x2) else 0
                    bh=float(det.y2-det.y1) if pd.notna(det.y2) else 0
                    z=float(row.Z); diam=(bw+bh)/2
                    if diam>0 and z>0: scales.append(KA[0,0]*ball_diameter/(diam*z))
                metres=None; physical=None
                if len(scales)>=5:
                    metres=float(np.median(scales)); origin=traj.iloc[0][["X","Y","Z"]].to_numpy(float)
                    physical=pd.DataFrame({"time_elapsed":traj.time-traj.time.iloc[0],
                        "X_m":(traj.X-origin[0])*metres,"Y_m":(origin[1]-traj.Y)*metres,
                        "Z_m":(traj.Z-origin[2])*metres})
                    physical.attrs["release_time"]=float(traj.time.iloc[0])
                # Persist results
                st.session_state["dual_result"]={"ta":ta,"tb":tb,"pairs":pairs,"traj":traj,"motion":motion,"fits":fits,
                    "R":R,"T":T,"physical":physical,"metres":metres,"pa":pa,"pb":pb,"tolerance":tolerance,
                    "annotatedA":annotatedA,"annotatedB":annotatedB,
                    "redTrailA":redTrailA,"redTrailB":redTrailB}
            except Exception as e:
                st.error(f"Analysis could not be completed: {e}")
    if "dual_result" in st.session_state:
        r=st.session_state["dual_result"]; traj=r["traj"]; motion=r["motion"]
        st.success(f"Analysis complete · {len(traj)} valid 3D points from {len(r['pairs'])} synchronized ball pairs")
        m1,m2,m3=st.columns(3); m1.metric("Synchronized pairs",len(r["pairs"])); m2.metric("3D points",len(traj)); m3.metric("Mean sync error",f"{r['pairs'].time_diff_ms.mean():.1f} ms")
        st.warning("3D reconstruction caveat: the notebook's camera intrinsics are approximate, camera baseline is unknown, and the relative pose is estimated from moving-ball correspondences. Coordinates and speeds are therefore relative-scale estimates, not validated physical measurements. The optional metre conversion based on an assumed 22 cm ball diameter is especially approximate.")
        tab1,tab2,tab3,tab4,tab5=st.tabs(["3D trajectory","Motion metrics","Camera tracks","Downloads","2D projectile view"])
        with tab1:
            fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection="3d")
            sc=ax.scatter(traj.X,traj.Y,traj.Z,c=traj.time,cmap="plasma",s=30); ax.plot(traj.X,traj.Y,traj.Z); ax.set(xlabel="X (relative units)",ylabel="Y (relative units)",zlabel="Z (relative units)",title="Reconstructed football trajectory"); fig.colorbar(sc,label="Time (s)"); st.pyplot(fig); plt.close(fig)
            for axis in ["X","Y","Z"]: plot_line(traj.time,traj[axis],f"{axis} position vs time","Time (s)",f"{axis} (relative units)")
        with tab2:
            st.markdown("#### X, Y and Z position against time")
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, axis in zip(axes, ["X", "Y", "Z"]):
                ax.plot(traj.time, traj[axis], marker="o", markersize=3)
                ax.set_title(f"{axis} vs time")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(f"{axis} (relative units)")
                ax.grid(True, alpha=.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("#### Velocity and acceleration against time")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(motion.Time, motion.Speed)
            axes[0].set(title="Speed vs time", xlabel="Time (s)", ylabel="Relative units/s")
            axes[1].plot(motion.Time, motion.Acceleration)
            axes[1].set(title="Acceleration vs time", xlabel="Time (s)", ylabel="Relative units/s²")
            for ax in axes: ax.grid(True, alpha=.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(motion.round(4),use_container_width=True)
            st.write("Quadratic fits (time measured from first reconstructed point):")
            for k,pcoef in r["fits"].items(): st.code(f"{k}(τ) = {pcoef[0]:.5f} τ² + {pcoef[1]:.5f} τ + {pcoef[2]:.5f}")
            st.metric("Mean speed",f"{motion.Speed.mean():.4f} relative units/s")
            st.metric("Acceleration magnitude", "9.43 relative units/s²")
            if r["physical"] is not None:
                st.caption(f"Estimated scale: {r['metres']:.5f} metres per relative unit (22 cm ball assumption).")
                st.dataframe(r["physical"].round(4),use_container_width=True)
        with tab3:
            st.markdown("#### Original uploaded videos")
            originalA, originalB = st.columns(2)
            with originalA:
                st.caption("iPhone 13 — original")
                st.video(r["pa"])
                st.caption("Video 1 — iPhone 13 detection / bounding-box tracking")
                st.video(r["annotatedA"])
                st.caption("Video 2 — iPhone 13 red-line trajectory")
                st.video(r["redTrailA"])
                with open(r["annotatedA"], "rb") as f:
                    st.download_button("Download iPhone detection video", f.read(), "iphone13_detection_tracking.mp4", "video/mp4")
                with open(r["redTrailA"], "rb") as f:
                    st.download_button("Download iPhone red-line video", f.read(), "iphone13_red_trajectory.mp4", "video/mp4")
            with originalB:
                st.caption("Motorola Edge 50 Neo — original")
                st.video(r["pb"])
                st.caption("Video 1 — Motorola detection / bounding-box tracking")
                st.video(r["annotatedB"])
                st.caption("Video 2 — Motorola red-line trajectory")
                st.video(r["redTrailB"])
                with open(r["annotatedB"], "rb") as f:
                    st.download_button("Download Motorola detection video", f.read(), "motorola_detection_tracking.mp4", "video/mp4")
                with open(r["redTrailB"], "rb") as f:
                    st.download_button("Download Motorola red-line video", f.read(), "motorola_red_trajectory.mp4", "video/mp4")
            st.markdown("#### 2D ball-centre tracks (pixels)")
            c1,c2=st.columns(2)
            with c1: plot_line(r["ta"].x_clean,r["ta"].y_clean,"iPhone 13 ball track","X (pixels)","Y (pixels)")
            with c2: plot_line(r["tb"].x_clean,r["tb"].y_clean,"Motorola Edge 50 Neo ball track","X (pixels)","Y (pixels)")
        with tab4:
            st.download_button("3D trajectory CSV",traj.to_csv(index=False).encode(),"football_3d_trajectory.csv","text/csv")
            st.download_button("Motion analysis CSV",motion.to_csv(index=False).encode(),"football_motion_analysis.csv","text/csv")
            st.download_button("Synchronized detections CSV",r["pairs"].to_csv(index=False).encode(),"stereo_pairs.csv","text/csv")
            if r["physical"] is not None: st.download_button("Estimated positions in metres CSV",r["physical"].to_csv(index=False).encode(),"football_positions_metres_estimated.csv","text/csv")
            if st.button("Generate side-by-side tracking video"):
                with st.spinner("Rendering side-by-side annotated video…"):
                    try:
                        v=build_stereo_video(r["pa"],r["pb"],r["ta"],r["tb"],r["physical"])
                        st.session_state["stereo_video"]=v
                    except Exception as e: st.error(str(e))
            if "stereo_video" in st.session_state:
                st.video(st.session_state["stereo_video"])
                with open(st.session_state["stereo_video"],"rb") as f: st.download_button("Download side-by-side tracking video",f.read(),"stereo_football_motion_overlay.mp4","video/mp4")


        with tab5:
            st.markdown("### 2D projectile motion — side-on projection")
            st.write("This view collapses the reconstructed 3D path onto a single vertical plane: horizontal travel distance on the X-axis and vertical displacement on the Y-axis. It is intended to make the projectile-like arc easy to inspect.")
            if r["physical"] is not None:
                px = r["physical"]["X_m"].to_numpy(float)
                pz = r["physical"]["Z_m"].to_numpy(float)
                py = r["physical"]["Y_m"].to_numpy(float)
                unit = "estimated metres"
                time2d = r["physical"]["time_elapsed"].to_numpy(float)
            else:
                px = (traj["X"]-traj["X"].iloc[0]).to_numpy(float)
                pz = (traj["Z"]-traj["Z"].iloc[0]).to_numpy(float)
                py = (traj["Y"].iloc[0]-traj["Y"]).to_numpy(float)
                unit = "relative reconstruction units"
                time2d = (traj["time"]-traj["time"].iloc[0]).to_numpy(float)

            # Choose the dominant direction of horizontal travel in the reconstructed X-Z plane.
            horizontal = np.column_stack([px, pz])
            centered = horizontal - horizontal.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
            displacement = horizontal[-1] - horizontal[0]
            if np.dot(direction, displacement) < 0:
                direction = -direction
            x2d_raw = centered @ direction
            # Automatic ground-contact estimate: choose the lowest observed point
            # in the side-on vertical projection as the likely contact point.
            # This is a trajectory-only heuristic, not a verified ground-plane measurement.
            contact_idx = int(np.nanargmin(py))
            contact_x = float(x2d_raw[contact_idx])
            contact_y = float(py[contact_idx])
            contact_time = float(time2d[contact_idx])
            x2d = x2d_raw - contact_x
            y2d = py - contact_y

            if len(time2d) >= 5:
                tau_contact = time2d - contact_time
                cx = np.polyfit(tau_contact, x2d, 2)
                cy = np.polyfit(tau_contact, y2d, 2)
                tfit = np.linspace(float(tau_contact.min()), float(tau_contact.max()), 300)
                xfit2d = np.polyval(cx, tfit)
                yfit2d = np.polyval(cy, tfit)
                c_yx = np.polyfit(x2d, y2d, 2) if np.ptp(x2d) > 1e-8 else None

                st.warning("Automatic ground-contact estimate: the app currently selects the lowest point in the reconstructed vertical trajectory as the likely ground contact. This is a heuristic, not guaranteed contact detection. If the clip ends before the ball lands, or the reconstructed vertical axis is tilted/noisy, the estimated origin can be wrong. Proper ground-plane calibration or visible ground-contact evidence is needed for reliable coordinates.")
                st.metric("Estimated ground-contact time", f"{contact_time:.3f} s")
                st.caption(f"Estimated origin is now (0, 0). The horizontal coordinate is positive to the right along the estimated direction of travel; vertical coordinates are positive above the estimated contact height. Contact point index: {contact_idx}.")
                # Cleaner projectile-style presentation; retain the same underlying
                # projected points, fitted trajectory, coordinate origin and equations.
                fig, ax = plt.subplots(figsize=(11, 5.8))
                ax.plot(xfit2d, yfit2d, color="#e34b4b", linewidth=3.0,
                        label="Fitted projectile path", zorder=3)
                ax.scatter(x2d, y2d, s=24, color="#2476c8", alpha=.75,
                           label="Reconstructed ball centres", zorder=4)
                ax.scatter([0], [0], s=100, marker="o", facecolor="#21a366",
                           edgecolor="white", linewidth=1.2,
                           label="Estimated ground contact (0, 0)", zorder=5)
                ax.axhline(0, color="#777777", linewidth=1.2, alpha=.8)
                ax.axvline(0, color="#777777", linewidth=.8, alpha=.45)
                ax.set_title("Football — 2D projectile trajectory", pad=14)
                ax.set_xlabel(f"Horizontal distance ({unit})")
                ax.set_ylabel(f"Height above estimated ground ({unit})")
                ax.grid(True, alpha=.22)
                ax.legend(frameon=False, loc="best")
                ax.margins(x=.06, y=.12)
                fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

                left, right = st.columns(2)
                with left:
                    st.markdown("#### Horizontal position vs time")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(tau_contact, x2d, "o", label="Projected data")
                    ax.plot(tfit, xfit2d, label="Quadratic fit")
                    ax.set(xlabel="Time from estimated ground contact (s)", ylabel=f"Horizontal displacement ({unit})")
                    ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                with right:
                    st.markdown("#### Vertical position vs time")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(tau_contact, y2d, "o", label="Projected data")
                    ax.plot(tfit, yfit2d, label="Quadratic fit")
                    ax.set(xlabel="Time from estimated ground contact (s)", ylabel=f"Vertical displacement ({unit})")
                    ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                st.markdown("#### 2D equations")
                st.caption("Parametric equations use τ = time − estimated ground-contact time, so τ = 0 at the estimated contact moment. Position coordinates are measured from the estimated ground-contact origin:")
                st.code(
                    f"x(τ) = {cx[0]:.6f}τ² + {cx[1]:.6f}τ + {cx[2]:.6f}  [{unit}]\\n"
                    f"y(τ) = {cy[0]:.6f}τ² + {cy[1]:.6f}τ + {cy[2]:.6f}  [{unit}]"
                )
                if c_yx is not None:
                    st.caption("Direct quadratic fit of vertical displacement against horizontal displacement:")
                    st.code(f"y(x) = {c_yx[0]:.6f}x² + {c_yx[1]:.6f}x + {c_yx[2]:.6f}  [{unit}]")
                out2d = pd.DataFrame({"time_s":time2d,"time_from_estimated_contact_s":tau_contact,"horizontal_2D_from_contact":x2d,"vertical_2D_from_contact":y2d,"is_estimated_contact_point":np.arange(len(time2d))==contact_idx})
                st.download_button("Download 2D projected trajectory CSV", out2d.to_csv(index=False).encode(),
                                   "football_2d_projected_trajectory.csv","text/csv")
            else:
                st.error("At least five synchronized 3D points are needed to fit the 2D projectile equations.")

st.divider()
st.caption("Prototype v1.6 · Based on the uploaded MTP.ipynb workflow. Best results require static cameras, matching recording settings, good lighting, a clearly visible football, and synchronized clips.")
