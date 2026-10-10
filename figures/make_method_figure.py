import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; fm.fontManager.addfont(fp)
plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name()
fig,ax=plt.subplots(figsize=(16,8.6),dpi=125); ax.set_xlim(0,160); ax.set_ylim(0,86); ax.axis("off")
F="#dbeafe";E="#1d4ed8";T="#fee2e2";R="#dc2626";G="#f3f4f6"
def box(x,y,w,h,t,fc,ec,fs=12,b=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,weight="bold" if b else "normal")
def arr(x1,y1,x2,y2,c="#374151",s="-|>",ls="-"):
    ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle=s,color=c,lw=2,ls=ls))
ax.text(80,83,"MM-OPD：无编码器学生 + 同族 VLM 教师在线策略蒸馏",ha="center",fontsize=20,weight="bold")
for x,t in [(2,"阶段 1：投影层热身（对齐）"),(62,"阶段 2：在线策略蒸馏（OPD）")]:
    ax.add_patch(FancyBboxPatch((x,4),(54 if x==2 else 94),72,boxstyle="round,pad=0.3",fc="#fafafa",ec="#9ca3af",lw=1.5,ls="--"))
    ax.text(x+2,73,t,fontsize=15,weight="bold",va="top")
# stage1
box(5,54,20,8,"图像 → 原始 patch\n(14×14 像素块)",G,"#6b7280",11)
box(5,38,20,8,"线性投影层【训】\n(唯一新增参数)",T,R,11,True)
box(30,38,22,8,"文本 token 嵌入\n(图文描述)",G,"#6b7280",11)
box(5,18,47,10,"Qwen3-4B（纯文本 LLM）【冻】",F,E,13,True)
arr(15,54,15,46.6); arr(15,38,22,28.6); arr(41,38,35,28.6)
box(5,6,47,7,"损失：图文对 next-token 交叉熵（SFT）",  "#fef9c3","#ca8a04",11)
arr(28,18,28,13.6)
ax.text(28,50,"少量图文对（如 ~0.5M）",ha="center",fontsize=10,color="#4b5563")
# stage2
box(65,56,26,8,"输入：图像 + 问题",G,"#6b7280",12)
box(65,40,26,8,"线性投影【训】 + Qwen3-4B\n+ LoRA【训】（学生）",T,R,11,True)
box(65,22,26,9,"学生自采样回答 y\n(vLLM 生成，GPU0)",G,"#6b7280",11)
box(118,40,26,8,"Qwen3-VL-8B 教师【冻】\n(bf16 推理，GPU1)",F,E,11,True)
box(110,22,42,9,"教师看：同一图像 + 问题 + 学生前缀 y<t\n→ 每个位置 next-token 分布 p_T",G,"#6b7280",10.5)
arr(78,56,78,48.6); arr(78,40,78,31.6)
arr(91,60,131,48.6,ls="--"); ax.text(112,57,"同一图像与问题",fontsize=10,color="#4b5563")
arr(91,26.5,109.4,26.5); ax.text(100,28,"y",fontsize=12,ha="center")
arr(131,40,131,31.6)
box(80,7,60,8,"损失：答案 token 上的逐 token 反向 KL\nL = Σ_t KL( p_S(·|x, y<t) ‖ p_T(·|img, x, y<t) )","#fef9c3","#ca8a04",11)
arr(78,22,95,15.6); arr(131,22,125,15.6)
ax.plot([79.6,63.3],[11,11],color=R,lw=2); arr(63.3,11,63.3,44,c=R); ax.text(64.6,17.2,"梯度只更新投影层 + LoRA",fontsize=10,color=R)
ax.text(146,10,"共享词表\n151,936",fontsize=10,ha="center",color=E)
ax.text(80,1,"图例：【训】/红色 = 可训练   【冻】/蓝色 = 冻结    虚线 = 教师输入    快速迭代版：Qwen3-1.7B 学生 + Qwen3-VL-4B 教师",ha="center",fontsize=11)
plt.savefig("/workspace/mm_opd_method.png",facecolor="white",bbox_inches="tight",pad_inches=0.2)
