# proxy_matcher_v22.py

# pip install faiss-cpu streamlit transformers rapidfuzz plotly scikit-learn



import sys, asyncio

# ─── STREAMLIT EVENT LOOP POLICY ────────────────────────────────────

# Fix “no running event loop” on Windows

if sys.platform.startswith("win"):

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



import torch

torch.classes.__path__ = []


import os

import pathlib

import re



import streamlit as st

st.set_page_config(page_title="🔍 Advanced Proxy Matcher v22 (FinBERT + Trading Hero LLM)", layout="wide")



import pandas as pd

import numpy as np

import faiss

from rapidfuzz import fuzz

from transformers import AutoTokenizer, AutoModel

from sklearn.preprocessing import normalize

import plotly.express as pex



# ─── THEME & CSS ────────────────────────────────────────────────────

ING_ORANGE    = "#FF6600"

ING_SHADE_MAP = {

    "BOND":      "#FF6600",

    "CDS":       "#FF8C00",

    "CR":        "#FF8C00",

    "FXSPOT":    "#FFA500",

    "FXVOL":     "#FFB74D",

    "IRSWAP":    "#FFC107",

    "IRSWVOL":   "#FFD54F",

    "IR_LINEAR": "#FFE082"

}

st.markdown(f"""

<style>

input[type="range"] {{ accent-color: {ING_ORANGE}; }}

.stButton>button {{

    background-color: {ING_ORANGE};

    color: white;

    font-weight: bold;

}}

.stSidebar .css-1d391kg h2 {{

    color: {ING_ORANGE};

}}

</style>

""", unsafe_allow_html=True)



# ─── HEADER & INSTRUCTIONS ─────────────────────────────────────────

st.title("🔍 Advanced Proxy Matcher v22 — Enterprise Edition")

st.markdown("""

**Powered by FinBERT & Trading Hero LLM embeddings**  

1. **Upload** your proxy & universe CSVs (one code per row, no header).  

2. **Auto-detect** asset classes & parse any `shock` fields.  

3. **Configure** business rules & per-field weights.  

4. **Set** **α** (hybrid vs semantic) & **top_k** ANN candidates.  

5. **Run**, inspect per-asset tabs & charts, and **download** your results.

""")



# ─── LOAD FINBERT/TRADING-HERO LLM ─────────────────────────────────

@st.cache_resource(show_spinner=False)

def load_finbert():

    tok = AutoTokenizer.from_pretrained("fuchenru/Trading-Hero-LLM")

    mdl = AutoModel.from_pretrained("fuchenru/Trading-Hero-LLM",

                                    output_hidden_states=True).eval()

    return tok, mdl



tokenizer, finbert = load_finbert()

DIM = finbert.config.hidden_size



def finbert_embed(texts: list[str]) -> np.ndarray:

    """Mean-pool last hidden state, then L2 normalize."""

    batches = []

    for i in range(0, len(texts), 32):

        chunk = texts[i:i+32]

        toks  = tokenizer(chunk, padding=True, truncation=True,

                          return_tensors="pt", max_length=128)

        with torch.no_grad():

            out    = finbert(**toks, output_hidden_states=True)

            last   = out.hidden_states[-1]                   # (B,S,H)

            mask   = toks.attention_mask.unsqueeze(-1)       # (B,S,1)

            summed = (last * mask).sum(1)                    # (B,H)

            counts = mask.sum(1).clamp(min=1)                # (B,1)

            embs   = (summed / counts).cpu().numpy()        # (B,H)

        batches.append(embs)

    all_embs = np.vstack(batches)

    return normalize(all_embs, axis=1)



# ─── FAISS INDEX BUILDERS ──────────────────────────────────────────

@st.cache_data

def embed_col(df: pd.DataFrame, col: str) -> np.ndarray:

    return finbert_embed(df[col].fillna("").astype(str).tolist())



@st.cache_resource

def build_ann_idx(df: pd.DataFrame, field: str):

    embs = embed_col(df, field).astype("float32")

    idx  = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)

    return idx, embs



@st.cache_resource

def build_fullcode_ann(df: pd.DataFrame):

    if df.empty:

        return faiss.IndexFlatIP(DIM), np.zeros((0, DIM), dtype="float32")

    embs = finbert_embed(df["original"].fillna("").tolist()).astype("float32")

    idx  = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)

    return idx, embs



# ─── GLOBAL CONFIG ─────────────────────────────────────────────────

REGION_GROUP = {

    "Northern America":"NA","Latin America":"LATAM",

    "Asia":"APAC","Oceania":"APAC",

    "Northwest Europe":"EMEA","Southern Europe":"EMEA","AroundAfrica":"EMEA"

}

RATING_SCORE = {

    "AAA":5,"AA+":4.5,"AA":4,"AA-":3.5,

    "A+":3,"A":2.5,"A-":2,

    "BBB+":1.5,"BBB":1,"BB":0.5,"B":0

}

ASSET_CONFIG = {

    "BOND":      {"fields":["rating","tenor","seniority","sector","region","currency","covered","shock"],

                  "weights":{"rating":3,"tenor":3,"seniority":2,"sector":1,"region":1,"currency":1,"covered":0.5,"shock":2},

                  "ann_field":None},

    "CDS":       {"fields":["instrument","issuer","currency","seniority","liquidity","maturity","shock"],

                  "weights":{"instrument":1,"issuer":3,"currency":2,"seniority":2,"liquidity":1,"maturity":1,"shock":2},

                  "ann_field":"issuer"},

    "CR":        {"fields":["instrument","issuer","currency","seniority","liquidity","maturity","shock"],

                  "weights":{"instrument":1,"issuer":3,"currency":2,"seniority":2,"liquidity":1,"maturity":1,"shock":2},

                  "ann_field":"issuer"},

    "FXSPOT":    {"fields":["spot","pair","shock"],

                  "weights":{"spot":1,"pair":1,"shock":2},

                  "ann_field":"pair"},

    "FXVOL":     {"fields":["smile","type","pair","strike","option_type","tenor","shock"],

                  "weights":{"smile":2,"type":2,"pair":2,"strike":1,"option_type":1,"tenor":1,"shock":2},

                  "ann_field":"pair"},

    "IRSWAP":    {"fields":["instrument","curve_name","currency","tenor","shock"],

                  "weights":{"instrument":1,"curve_name":3,"currency":1,"tenor":1,"shock":2},

                  "ann_field":"curve_name"},

    "IRSWVOL":   {"fields":["method","currency","tenor","shock"],

                  "weights":{"method":2,"currency":1,"tenor":1,"shock":2},

                  "ann_field":"method"},

    "IR_LINEAR": {"fields":["instrument","curve_name","rate","currency","tenor","shock"],

                  "weights":{"instrument":1,"curve_name":3,"rate":2,"currency":1,"tenor":1,"shock":2},

                  "ann_field":"curve_name"},

    # alias IRFCVOL → use same config as IRSWVOL

    "IRFCVOL":   {"fields":["method","currency","tenor","shock"],

              "weights":{"method":2,"currency":1,"tenor":1,"shock":2},

              "ann_field":"method"}

}



# ─── ING LOGO ───────────────────────────────────────────────────────

logo_paths = [

    pathlib.Path(__file__).resolve().parent / "Inglogo.jpg",

    pathlib.Path.cwd() / "Inglogo.jpg",

    pathlib.Path(__file__).resolve().parent / "proxy" / "Inglogo.jpg",

    pathlib.Path.cwd() / "proxy" / "Inglogo.jpg",

]

logo_file = next((p for p in logo_paths if p.exists()), None)

if logo_file:

    st.sidebar.image(str(logo_file), width=140)

else:

    st.sidebar.warning("⚠️ ING logo missing—please place Inglogo.jpg beside this script or in proxy/")



# ─── HELPERS & MATCHING ────────────────────────────────────────────

def tenor_to_months(t: str) -> float:

    s = str(t or "").strip().upper()

    if s=="ON": return 1/30

    m = re.match(r"^(\d+(?:\.\d+)?)([DWMY])$", s)

    if not m: return 0.0

    v,u = float(m.group(1)), m.group(2)

    return {"D":1/30,"W":7/30,"M":1,"Y":12}[u] * v



def rating_dist(a,b):

    return 1 - abs(RATING_SCORE.get(a,0)-RATING_SCORE.get(b,0))/5



def region_score(a,b):

    if not a or not b: return 0

    return 1 if a==b or REGION_GROUP.get(a)==REGION_GROUP.get(b) else 0



def hybrid_score(px, cd, cfg):

    tot, wsum = 0.0, 0.0

    for f,w in cfg["weights"].items():

        p, c = px.get(f,""), cd.get(f,"")

        if f=="rating":

            sc=rating_dist(p,c)

        elif f in ("tenor","maturity"):

            a,b=tenor_to_months(p),tenor_to_months(c)

            sc=1-abs(a-b)/max(a,b,1)

        elif f=="region":

            sc=region_score(p,c)

        elif f=="shock":

            sc=1.0 if p and p==c else 0.0

        else:

            sc=1.0 if p==c else fuzz.partial_ratio(str(p),str(c))/100

        tot+=w*sc; wsum+=w

    return (tot/wsum) if wsum else 0.0



def parse_row(code: str) -> dict:

    parts=[p.strip() for p in code.split(":")]

    if parts[0].upper()=="CR": parts=parts[1:]

    a0=parts[0].upper()

    if a0 in ("CDS","CR"):       asset="CDS"

    elif a0=="BOND":             asset="BOND"

    elif a0=="FXSPOT":           asset="FXSPOT"

    elif a0=="FXVOL":            asset="FXVOL"

    elif a0=="IR" and len(parts)>1 and parts[1].upper()=="SWAP" and len(parts)==5:

                                  asset="IRSWAP"

    elif a0=="IR" and len(parts)>1 and parts[1].upper() in ("SWVOL","SWAPVOL"):

                                  asset="IRSWVOL"

    elif a0=="IR":               asset="IR_LINEAR"

    else:                        asset=a0

    out={"asset":asset,"original":code}

    schema={

      "BOND":      ["rating","tenor","seniority","sector","region","currency","covered","shock"],

      "CDS":       ["instrument","issuer","currency","seniority","liquidity","maturity","shock"],

      "FXSPOT":    ["spot","pair","shock"],

      "FXVOL":     ["smile","type","pair","strike","option_type","tenor","shock"],

      "IRSWAP":    ["instrument","curve_name","currency","tenor","shock"],

      "IRSWVOL":   ["method","currency","tenor","shock"],

      "IR_LINEAR": ["instrument","curve_name","rate","currency","tenor","shock"]

    }

    for i,k in enumerate(schema.get(asset,[]), start=1):

        out[k]=parts[i] if i<len(parts) else ""

    return out



def parse_df(df: pd.DataFrame)->pd.DataFrame:

    return pd.json_normalize(df["code"].astype(str).apply(parse_row))



def apply_business_rules(px, univ, rules):

    df = univ.copy()

    r  = rules.get(px["asset"],{})

    for fld in r.get("require_exact",[]):

        df = df[df[fld]==px.get(fld,"")]

    if r.get("prefer"):

        df["_boost"] = df.apply(lambda row: sum(

            0.1 for fld,vals in r["prefer"].items() if row[fld] in vals

        ), axis=1)

    else:

        df["_boost"]=0.0

    return df



def two_stage_match(px, univ, cfg, rules, α, top_k):

    df = apply_business_rules(px, univ, rules)

    if df.empty:

        df = univ.copy().assign(_boost=0.0)

    ann0, fe = build_fullcode_ann(df)

    q         = finbert_embed([px["original"]]).astype("float32")

    _, ids    = ann0.search(q, top_k)

    cand      = df.iloc[ids[0]].reset_index(drop=True)

    h_scores  = cand.apply(lambda r: hybrid_score(px, r, cfg), axis=1).values

    s_scores  = (fe[ids[0]] @ q[0]).flatten() if fe.shape[0]>0 else np.zeros_like(h_scores)

    final     = α*h_scores + (1-α)*s_scores + cand["_boost"].values

    best_idx  = int(final.argmax())

    return cand.loc[best_idx,"original"], float(final[best_idx])



# ─── CONTROLS & RUN ──────────────────────────────────────────────────

alpha = st.sidebar.slider("Blend α (hybrid vs semantic)", 0.0, 1.0, 0.6, 0.05)

top_k = st.sidebar.slider("ANN top_k candidates",  1, 50, 10, 1)



c1, c2 = st.columns(2)

with c1: proxy_u = st.file_uploader("📄 Proxy CSV",   type="csv")

with c2: univ_u  = st.file_uploader("📄 Universe CSV",type="csv")



if proxy_u and univ_u:

    px_raw = pd.read_csv(proxy_u, header=None, names=["code"])

    un_raw = pd.read_csv(univ_u,   header=None, names=["code"])

    px_df  = parse_df(px_raw)

    un_df  = parse_df(un_raw)



    detected=sorted(px_df["asset"].unique())

    st.sidebar.header("🔧 Detected Asset Classes")

    sel=st.sidebar.multiselect("Which to match:",detected,default=detected)



    # Business-Rules UI

    rules={}

    for a in sel:

        if a not in ASSET_CONFIG:

            st.sidebar.error(f"No config for '{a}'"); continue

        with st.sidebar.expander(f"{a} Business Rules",expanded=False):

            fields=[f for f in ASSET_CONFIG[a]["fields"] if f!="shock"]

            req=st.multiselect(f"{a}: require exact on",fields,default=[])

            prefer={}

            for fld in fields:

                vals=un_df[un_df.asset==a][fld].dropna().unique().tolist()

                pv=st.multiselect(f"{a}: prefer {fld}",vals,default=[])

                if pv: prefer[fld]=pv

            rules[a]={"require_exact":req,"prefer":prefer}



    # Weights UI

    cfgs={}

    for a in sel:

        if a not in ASSET_CONFIG: continue

        cfg=ASSET_CONFIG[a].copy()

        with st.sidebar.expander(f"{a} Field Weights",expanded=False):

            for fld in cfg["fields"]:

                cfg["weights"][fld]=st.slider(f"{a} · {fld}",0.0,5.0,float(cfg["weights"][fld]),0.1)

        cfgs[a]=cfg



    # Run Matching

    if st.button("🔎 Run Matching"):

        with st.spinner("Running advanced matching…"):

            out=[]

            for a,cfg in cfgs.items():

                sub_px=px_df[px_df["asset"]==a]

                sub_un=un_df[un_df["asset"]==a].reset_index(drop=True)

                for _,px in sub_px.iterrows():

                    bp,sc=two_stage_match(px,sub_un,cfg,rules,alpha,top_k)

                    out.append({"asset":a,"proxy":px["original"],"best_proxy":bp,"score":round(sc,4),"formula":f"SV({bp})"})

        res=pd.DataFrame(out)

        st.success(f"✅ Matched {len(res)} proxies")



        # Global download

        _, colB = st.columns([3,1])

        colB.download_button("📥 Download All",

            data=res.to_csv(index=False).encode("utf-8"),

            file_name="matched_all.csv", mime="text/csv"

        )



        # Per-asset tabs

        tabs = st.tabs(sel)

        for i,a in enumerate(sel):

            df_a = res[res["asset"]==a].reset_index(drop=True)

            with tabs[i]:

                st.subheader(f"{a} — {len(df_a)} proxies")

                st.download_button(f"📥 Download {a}",

                    data=df_a.to_csv(index=False).encode("utf-8"),

                    file_name=f"matched_{a}.csv", mime="text/csv"

                )

                st.dataframe(df_a, height=200)



                fig_s = pex.scatter(

                    df_a, x=df_a.index, y="score",

                    color_discrete_sequence=[ING_SHADE_MAP[a]],

                    hover_data=["proxy","best_proxy","score"],

                    labels={"x":"","score":"Match Score"},

                    title=f"{a}: Match Scores"

                )

                fig_s.update_layout(xaxis=dict(showticklabels=False), margin=dict(t=40,b=20))

                fig_s.update_traces(marker=dict(size=10, opacity=0.7))

                st.plotly_chart(fig_s, use_container_width=True)



                fig_b = pex.box(

                    df_a, y="score",

                    color_discrete_sequence=[ING_SHADE_MAP[a]],

                    labels={"score":"Match Score"},

                    title=f"{a}: Score Distribution"

                )

                st.plotly_chart(fig_b, use_container_width=True)