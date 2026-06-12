#!/usr/bin/env python3
"""PaddleOCR-VL handwriting benchmark (CER/WER + IoU + reading order)."""
import json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

IMAGES_DIR = Path("/data/handwritten")
GT_PATH = Path("/data/ground_truth_handwritten.json")
RESULTS_DIR = Path("/results")
CANDIDATE = "paddleocr_vl_handwritten"

# --- metrics -----------------------------------------------------------
def _ed(s1, s2):
    if len(s1) < len(s2): return _ed(s2, s1)
    if not s2: return len(s1)
    prev = list(range(len(s2)+1))
    for c1 in s1:
        cur = [prev[0]+1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j+1]+1, cur[j]+1, prev[j]+(c1!=c2)))
        prev = cur
    return prev[-1]

def _norm(t): return re.sub(r"\s+"," ",t.replace("\n"," ").replace("\r"," ")).strip()

def cer(r,h):
    r=h=_norm(r);h=_norm(h)
    return _ed(r,h)/len(r) if r else (1.0 if h else 0.0)

def wer(r,h):
    rw=_norm(r).split(); hw=_norm(h).split()
    return _ed(rw,hw)/len(rw) if rw else (1.0 if hw else 0.0)

def _xyxy(b):
    if len(b)!=4: return [0,0,0,0]
    if b[2]<b[0] or b[3]<b[1]: return [b[0],b[1],b[0]+b[2],b[1]+b[3]]
    return list(b)

def _iou(a,b):
    xa=max(a[0],b[0]); ya=max(a[1],b[1]); xb=min(a[2],b[2]); yb=min(a[3],b[3])
    inter=max(0,xb-xa)*max(0,yb-ya)
    ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    denom=ua+ub-inter
    return inter/denom if denom>0 else 0.0

def block_iou(pred, gt, thr=0.1):
    if not gt: return {"mean_iou": 1.0 if not pred else 0.0, "matched": 0, "total_gt": 0, "total_pred": len(pred), "recall": 0.0, "precision": 0.0}
    matched=set(); ious=[]
    for p in pred:
        pb=_xyxy(p.get("bbox",[])); bi,bj=0.0,-1
        for j,g in enumerate(gt):
            if j in matched: continue
            i=_iou(pb,_xyxy(g.get("bbox",[])))
            if i>bi: bi,bj=i,j
        if bj>=0 and bi>=thr: ious.append(bi); matched.add(bj)
    n=len(ious); ng=len(gt); np=len(pred)
    return {"mean_iou": sum(ious)/n if n else 0.0, "matched": n, "total_gt": ng, "total_pred": np, "recall": n/ng if ng else 0.0, "precision": n/np if np else 0.0}

def reading_order(pred, gt, gt_order):
    if not pred: return {"kendall_tau": 0.0}
    idx=[(i,_xyxy(b.get("bbox",[0,0,0,0]))[1],_xyxy(b.get("bbox",[0,0,0,0]))[0]) for i,b in enumerate(pred)]
    idx.sort(key=lambda x:(x[1],x[2]))
    porder=[i for i,_,_ in idx]
    ranks=[]; matched=set()
    for pi in porder:
        pb=_xyxy(pred[pi].get("bbox",[])); bi,bj=0.0,-1
        for j in range(len(gt)):
            if j in matched: continue
            i=_iou(pb,_xyxy(gt[j].get("bbox",[])))
            if i>bi: bi,bj=i,j
        if bj>=0 and bi>0.05: ranks.append(bj); matched.add(bj)
    if not ranks or not gt_order or len(gt_order)!=len(gt): return {"kendall_tau": 0.0}
    gt_ranks=[gt_order[g] for g in ranks]
    pp=list(range(len(ranks)))
    sp=sorted(range(len(gt_ranks)), key=lambda i: gt_ranks[i])
    n=len(pp); conc=disc=0
    for i in range(n):
        for j in range(i+1,n):
            a=(pp[i]-pp[j])*(sp[i]-sp[j])
            if a>0: conc+=1
            elif a<0: disc+=1
    tot=conc+disc
    return {"kendall_tau": (conc-disc)/tot if tot>0 else 0.0}

# --- parse --------------------------------------------------------------
def parse(output):
    blocks=[]; text=""
    if output:
        for res in output:
            if hasattr(res,"json") and res.json:
                inner=res.json.get("res",res.json) if isinstance(res.json,dict) else {}
                for item in inner.get("parsing_res_list",[]):
                    t=item.get("block_content",""); br=item.get("block_bbox","")
                    if t:
                        if isinstance(br,list): bbox=[float(x) for x in br[:4]]
                        elif isinstance(br,str):
                            try: bbox=[int(float(x.strip())) for x in br.strip("[]").split(",")]
                            except: bbox=[0,0,0,0]
                        else: bbox=[0,0,0,0]
                        blocks.append({"bbox": bbox if len(bbox)==4 else[0,0,0,0], "text": str(t)})
                        text+=str(t)+" "
            if not text and hasattr(res,"markdown") and res.markdown:
                md=res.markdown; text=md.get("markdown_texts",str(md)) if isinstance(md,dict) else str(md)
            if not text and hasattr(res,"text") and res.text: text=str(res.text)
    return text.strip(), blocks

# --- main ---------------------------------------------------------------
def main():
    images=sorted(p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in {".png",".jpg",".jpeg"})
    if not images: print("ERROR: no images",file=sys.stderr); return 1
    gt_data={}
    if GT_PATH.exists():
        gt_data={e["image"]:e for e in json.loads(GT_PATH.read_text())}
    print(f"Images: {len(images)}  GT: {len(gt_data)}")

    from paddleocr import PaddleOCRVL
    import paddle
    pipe = PaddleOCRVL(use_doc_orientation_classify=False, use_doc_unwarping=False)
    gpu = paddle.device.cuda.get_device_name(0)
    print(f"Pipeline loaded. GPU: {gpu}")

    # warmup
    print(f"Warmup: {images[0].name} ...")
    paddle.device.synchronize()
    pipe.predict(str(images[0]))
    paddle.device.synchronize()
    print("Warmup done.\n")

    lats=[]; texts=[]; blocks=[]; names=[]; ious=[]; taus=[]
    for i,img in enumerate(images):
        fn=img.name
        paddle.device.synchronize()
        t0=time.perf_counter()
        out=pipe.predict(str(img))
        paddle.device.synchronize()
        el=time.perf_counter()-t0
        txt,blk=parse(out)
        lats.append(el); texts.append(txt); blocks.append(blk); names.append(fn)

        gt=gt_data.get(fn,{})
        gt_blk=gt.get("blocks",[]); gt_ord=gt.get("reading_order",[])
        io={}; ta={}
        if gt_blk and blk:
            io=block_iou(blk, gt_blk)
            ta=reading_order(blk, gt_blk, gt_ord)
        elif gt_blk:
            io={"mean_iou":0,"matched":0,"total_gt":len(gt_blk),"total_pred":0,"recall":0,"precision":0}
            ta={"kendall_tau":0}
        ious.append(io); taus.append(ta)
        print(f"  [{i+1:2d}/{len(images)}] {fn}: {el:.2f}s  IoU={io.get('mean_iou',0):.2f}  tau={ta.get('kendall_tau',0):.2f}")

    n=len(lats); al=sum(lats)/n
    sl=(sum((x-al)**2 for x in lats)/(n-1))**0.5 if n>1 else 0
    csv=[cer(gt_data.get(nm,{}).get("text",""),t) for nm,t in zip(names,texts)]
    wsv=[wer(gt_data.get(nm,{}).get("text",""),t) for nm,t in zip(names,texts)]
    ac=sum(csv)/len(csv) if csv else 0; aw=sum(wsv)/len(wsv) if wsv else 0
    vi=[r for r in ious if r]; ai=sum(r["mean_iou"] for r in vi)/len(vi) if vi else 0
    ar=sum(r.get("recall",0) for r in vi)/len(vi) if vi else 0
    ap=sum(r.get("precision",0) for r in vi)/len(vi) if vi else 0
    vt=[r["kendall_tau"] for r in taus if r]; at=sum(vt)/len(vt) if vt else 0

    result={
        "candidate_name": CANDIDATE, "model_version": "PaddleOCR-VL-1.6-0.9B",
        "timestamp": datetime.now(timezone.utc).isoformat(), "gpu_name": gpu,
        "latency_total_avg": al, "latency_total_std": sl, "cer": ac, "wer": aw,
        "reading_order_tau": at, "bounding_box_iou": ai,
        "throughput_ppm": 60/al if al>0 else 0,
        "iou_details": {"mean_iou": ai, "mean_recall": ar, "mean_precision": ap},
        "reading_order_details": {"mean_kendall_tau": at},
        "notes": "PaddleOCR-VL-1.6 handwriting benchmark. Docker sm120-offline."
    }
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    (RESULTS_DIR/f"{CANDIDATE}.json").write_text(json.dumps(result,indent=2))
    print(f"\nSaved: {RESULTS_DIR}/{CANDIDATE}.json")
    print(f"CER={ac:.4f} WER={aw:.4f} IoU={ai:.4f} tau={at:.4f} Lat={al:.2f}s")
    return 0

if __name__=="__main__":
    sys.exit(main())
