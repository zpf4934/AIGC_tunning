# -*- coding:utf-8 -*-
"""
# File       : web_demo.py
# Time       ：2023/7/17 10:46
# Author     ：andy
# version    ：python 3.9
"""
import os
import gradio as gr
import mdtex2html
import torch

from predict import load_model, generate

css = """
#chuanhu_chatbot {
    transition: height 0.3s ease;
}

#app_title {
    font-weight: var(--prose-header-text-weight);
    font-size: var(--text-xxl);
    line-height: 1.3;
    text-align: left;
    margin-top: 6px;
    white-space: nowrap;
}

#status_display {
    display: flex;
    min-height: 2em;
    align-items: flex-end;
    justify-content: flex-end;
}
#status_display p {
    font-size: .85em;
    font-family: ui-monospace, "SF Mono", "SFMono-Regular", "Menlo", "Consolas", "Liberation Mono", "Microsoft Yahei UI", "Microsoft Yahei", monospace;
    /* Windows下中文的monospace会fallback为新宋体，实在太丑，这里折中使用微软雅黑 */
    color: var(--body-text-color-subdued);
}

#status_display {
    transition: all 0.6s;
}
"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history, model, tokenizer):
    chatbot.append((parse_text(input), ""))
    result = generate(input, model, tokenizer, history=history, top_p=top_p, temperature=temperature, max_length=max_length)
    for response, history in result:
        chatbot[-1] = (parse_text(input), parse_text(response))
        yield chatbot, history


def get_model(model_select_dropdown):
    global model, tokenizer
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    model, tokenizer = load_model(model_select_dropdown)
    return model, tokenizer


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


models = os.listdir('outputs')
models.insert(0, '')
with gr.Blocks(css=css) as demo:
    model = gr.State()
    tokenizer = gr.State()
    with gr.Row():
        gr.HTML("AIGC", elem_id="app_title")
        status_display = gr.Markdown("信息", elem_id="status_display")
    with gr.Row().style(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row():
                chatbot = gr.Chatbot(label="Andy Chat", elem_id="chuanhu_chatbot").style(height="100%")
            with gr.Row():
                with gr.Column(min_width=225, scale=12):
                    user_input = gr.Textbox(
                        elem_id="user_input_tb",
                        show_label=False, placeholder="在这里输入"
                    ).style(container=False)
                with gr.Column(min_width=42, scale=1):
                    submitBtn = gr.Button(value="提交", variant="primary")
                with gr.Column(min_width=42, scale=1):
                    emptyBtn = gr.Button(value="清空")
        with gr.Column(scale=1):
            model_select_dropdown = gr.Dropdown(
                label="选择模型", choices=models, multiselect=False, value=models[0], interactive=True
            )
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, model, tokenizer],
                    [chatbot, history], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
    model_select_dropdown.change(get_model, [model_select_dropdown], [model, tokenizer])

demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0', server_port=6007, debug=True)
