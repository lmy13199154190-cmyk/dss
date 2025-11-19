from openai import OpenAI
import json
import pandas as pd
import sklearn as sk
import altair as alt
import numpy as np
from sklearn import datasets,metrics,tree,model_selection
from lightgbm import LGBMClassifier
import streamlit as st

api_key ="sk-UhMG3S62qNo9HNz2AbD6B0163e1149299a9cE3058e7f25Fc"
api_base ="https://maas-api.cn-huabei-1.xf-yun.com/v1"
MODEL_ID = "xop3qwen1b7"
client = OpenAI(api_key = api_key,base_url = api_base)

def ask_ai(messages,json_type = True,model_id = MODEL_ID):
    json_messages = [{"role":"user","content":messages}]
    extra_body = {"response_format": {"type": "json_object"}, "search_disable": True} if json_type else {}
    response = client.chat.completions.create(model=model_id, messages=json_messages, extra_body=extra_body)
    message = response.choices[0].message.content
    if json_type:
        try:
            return json.loads(message)
        except Exception:
            return {"error":"JSON格式解析失败","raw":message}
    return message



def ai_explain(task, method, ds_name, highlights):
    prompt = f"""
你是数据科学助教。请用中文简要解读下面的模型结果，并给出3-5条面向管理者的可执行建议（使用•项目
符号，不要输出代码）。
任务：{task}；方法：{method}；数据集：{ds_name}
关键结果：{highlights}
请先用1-2句话说明结果意味着什么，再给出建议；尽量避免术语，聚焦业务含义。
"""
    return ask_ai(prompt, json_type=False)

def load_dataset(task,ds_name):
    if ds_name==("Iris"):
        d = datasets.load_iris()
    elif ds_name==("Wine"):
        d = datasets.load_wine()
    else:
        d = datasets.load_breast_cancer()
    return d.data, d.target,d.feature_names,list(d.target_names)

def train_model(X,y,method):
        X_tr,X_te,y_tr,y_te = model_selection.train_test_split(X,y,test_size = 0.2 ,random_state =0)
        if method =="DecisionTree":
            model = tree.DecisionTreeClassifier(random_state = 0)
        else:
            model = LGBMClassifier(random_state = 0)
        model.fit(X_tr,y_tr)
        y_pred = model.predict(X_te)
        acc = metrics.accuracy_score(y_te, y_pred)
        cm = metrics.confusion_matrix(y_te, y_pred)
        return acc,cm,model

def plot_confusion_matrix(cm,target_names):
        cm_df = pd.DataFrame(
            cm,
            index=[f"T_{t}" for t in target_names],
            columns=[f"P_{t}" for t in target_names])
        chart = alt.Chart(cm_df.reset_index().melt("index")).mark_rect().encode(
            x=alt.X("variable:N"),
            y=alt.Y("index:N"),
            color=alt.Color("value:Q", title="Count")
        ).properties(title="Confusion Matrix")
        return chart


st.set_page_config(page_title="基于LLM的DSS系统", layout="wide")
st.title("📊 基于LLM的决策支持系统（DSS）原型")

# --------- 左侧栏 ---------
st.sidebar.header("任务与模型设置")
task_type = st.sidebar.selectbox("任务类型", ["分类"])
dataset_name = st.sidebar.selectbox("数据集", ["Iris", "Wine", "Breast Cancer"])
model_choice = st.sidebar.selectbox("模型选择", ["DecisionTree", "LightGBM"])

# --------- JSON DSS 模型信息 ---------
st.header("① 决策支持系统模型类型（AI生成JSON）")
messages = """
  请帮我整理下决策支持系统有哪些常见的模型类型，
  返回json结构，包含名称，适用问题，边界条件。
  输出结构如下
  {
  'system 1': {'name': XXX, 'question_type': XXX, 'boundary': XXX},
  'system 2': {'name': XXX, 'question_type': XXX, 'boundary': XXX},
  ...
  }
  """
if st.button("生成模型类型 JSON"):
    res = ask_ai(messages)
    st.dataframe(pd.DataFrame(res))

# --------- 模型训练与AI解读 ---------
st.header("② 模型训练与 AI 决策解读")

if st.button("开始训练模型"):
    X, y, features, targets = load_dataset(task_type, dataset_name)
    acc, cm, model = train_model(X, y, model_choice)
    st.metric("Accuracy", f"{acc:.3f}")
    st.altair_chart(plot_confusion_matrix(cm, targets), use_container_width=True)

    highlights = f"Accuracy={acc:.3f}；混淆矩阵规模={cm.shape}"
    explanation = ai_explain(task_type, model_choice, dataset_name, highlights)
    st.subheader("AI解读与管理建议")
    st.write(explanation)