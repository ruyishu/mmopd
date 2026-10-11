# extract jpgs from downloaded shards (idempotent); run repeatedly
import glob,os,pyarrow.parquet as pq,sys
out="data/coco/train2017"; n=0
done=set(open("data/coco/extracted.txt").read().split()) if os.path.exists("data/coco/extracted.txt") else set()
for f in sorted(glob.glob("data/coco/raw/det/data/train-*.parquet")):
    if f in done: continue
    pf=pq.ParquetFile(f)
    for b in pf.iter_batches(columns=["image_id","image"],batch_size=256):
        for iid,im in zip(b.column("image_id").to_pylist(),b.column("image").to_pylist()):
            p=f"{out}/{int(iid):012d}.jpg"
            if not os.path.exists(p): open(p,"wb").write(im["bytes"]); n+=1
    open("data/coco/extracted.txt","a").write(f+"\n"); print("extracted",f,flush=True)
print("new",n,"total",len(os.listdir(out)))
