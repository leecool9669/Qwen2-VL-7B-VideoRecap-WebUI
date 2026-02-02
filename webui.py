import gradio as gr

# 占位的模型加载函数，仅用于展示接口形态，不实际下载权重

def load_model():
    """返回一个伪模型对象，以 Qwen2-VL-7B 为叙述背景。"""

    class DummyModel:
        def __call__(self, video_path: str, text_prompt: str):
            return {
                "recap": "【演示结果】该描述以 Qwen2-VL-7B 为多模态基座，对视频内容进行通用化 Recap。",
                "key_frames": [
                    "基座片段 1：多模态注意力关注到人物与场景交互。",
                    "基座片段 2：时间维度上的长距离依赖被显式建模。",
                    "基座片段 3：全局信息被压缩为简洁自然语言摘要。",
                ],
            }

    return DummyModel()


model = load_model()


def analyze_video(video, prompt):
    if video is None:
        return "请先上传一段用于多模态基座示意的视频片段。", ["尚未检测到关键帧。"], ""

    outputs = model(str(video), prompt or "请从多模态表征学习的角度描述该视频。")
    recap_text = outputs["recap"]
    key_frames = outputs["key_frames"]
    key_frames_markdown = "\n".join(f"- {item}" for item in key_frames)
    return recap_text, key_frames, key_frames_markdown


with gr.Blocks(title="Qwen2-VL-7B VideoRecap WebUI（演示版）") as demo:

    gr.Markdown(
        """# Qwen2-VL-7B VideoRecap WebUI（演示版）\n\n"
        "用于展示多模态基座模型在长视频理解任务中的界面与数据流设计，可与 Tarsier2-Recap 系列模型形成对比。"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 输入区：多模态视频样本")
            video_input = gr.Video(label="上传用于基座示意的视频", sources=["upload"], interactive=True)
            prompt_input = gr.Textbox(
                label="文本指令（可选）",
                value="请从多模态对齐与表征学习角度总结该视频。",
                lines=3,
            )
            run_btn = gr.Button("开始分析（演示，不进行真实推理）", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. 结果区：基座视角 Recap")
            recap_output = gr.Textbox(
                label="视频长篇描述（基座视角示意输出）",
                lines=10,
                interactive=False,
            )
            keyframe_gallery = gr.HighlightedText(
                label="关键片段（按表征角色划分）",
                combine_adjacent=True,
            )

    with gr.Accordion("可选：Markdown 结果导出", open=False):
        keyframe_md = gr.Markdown(
            "尚未检测到关键帧。可将伪结果复制至模型设计文档。"
        )

    def _wrapped_analyze(video, prompt):
        recap, key_frames, key_md = analyze_video(video, prompt)
        highlighted = [(kf, "关键片段") for kf in key_frames]
        return recap, highlighted, key_md

    run_btn.click(
        _wrapped_analyze,
        inputs=[video_input, prompt_input],
        outputs=[recap_output, keyframe_gallery, keyframe_md],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863, show_error=True)
