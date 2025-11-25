import gradio as gr
from GIS_map import OUTCOME_METRICS, plot_outcome_map, COUNTIES
from application import TOP_K_DEFAULT, top_queries, init_engine, ask


# Build Gradio UI
def ui():
    with gr.Blocks() as demo:
        gr.HTML("""
        <style>
          body { 
              background-color: #1e1e1e !important; 
              color: #e8e8e8 !important;
          }

          .main-title { 
              font-size: 32px; 
              font-weight: 700; 
              margin-bottom: 10px; 
              color: #ffffff;
          }

          .section-box {
              background: #2a2a2a;
              padding: 18px;
              border-radius: 10px;
              border: 1px solid #3a3a3a;
              box-shadow: 0 2px 6px rgba(0,0,0,0.4);
              margin-bottom: 18px;
          }

          /* The answer area */
          .answer-box {
              background: #2f3136;
              border-radius: 10px;
              padding: 20px;
              font-size: 18px;
              line-height: 1.6;
              border: 1px solid #3a3a3a;
              min-height: 200px;
              color: #ffffff;
              white-space: pre-wrap;
          }

          /* Placeholder text */
          .answer-box em {
              color: #9aa0a6;
          }

          /* Fix Gradio dark text issues */
          label, .gradio-container * {
              color: #e8e8e8 !important;
          }

          input, textarea, select {
              background-color: #2a2a2a !important;
              color: #ffffff !important;
              border: 1px solid #3a3a3a !important;
          }

          .gr-button {
              background-color: #4b5563 !important;
              color: white !important;
              border-radius: 8px !important;
              border: none !important;
          }

          .gr-button:hover {
              background-color: #6b7280 !important;
          }
        </style>
        """)
        gr.Image(
            "image/cdss-logo.png",
            show_label=False,
            width=500)
        gr.HTML("<div class='main-title'>CalWORKs County QA System</div>")

        with gr.Group(elem_classes="section-box"):
            with gr.Row():
                llm_model = gr.Dropdown(
                    ["mistral:7b", "gpt-3.5-turbo"],
                    value="mistral:7b",
                    label="LLM Model"
                )

        with gr.Group(elem_classes="section-box"):
            with gr.Row():
                qbox = gr.Textbox(label="Question", scale=3)
                go = gr.Button("Answer", scale=1)

        answer = gr.HTML(
            "<div class='answer-box'><em>Answer will appear here...</em></div>",
            label="Answer"
        )

        go.click(
            lambda question, model: ask(question, 5,
                "BAAI" if model.startswith("mistral") else "OpenAI Embeddings",
                "BAAI/bge-m3" if model.startswith("mistral") else "text-embedding-3-large",
                "Ollama" if model.startswith("mistral") else "OpenAI",
                model),
            inputs=[qbox, llm_model],
            outputs=answer
        )


        with gr.Group(elem_classes="section-box"):
            gr.Markdown("### Top Queries")
            gr.Textbox(value=top_queries(), interactive=False, lines=10, label="")

        with gr.Group(elem_classes="section-box"):
            gr.Markdown("### County Outcome Map")
            metric_dd = gr.Dropdown(
                choices=OUTCOME_METRICS,
                value=OUTCOME_METRICS[0],
                label="Metric")
            map_plot = gr.Plot(label="Map")
            demo.load(plot_outcome_map,
                      inputs=[metric_dd],
                      outputs=map_plot)
            metric_dd.change(plot_outcome_map,
                             inputs=[metric_dd],
                             outputs=map_plot)

        gr.Image(
            "image/calworks_logo.jpeg",
            show_label=False,
            width=1600)
        placeholder = gr.Textbox(visible=False)

        demo.load(
            fn=lambda: init_engine(
                "BAAI",
                "BAAI/bge-m3",
                "Ollama",
                "mistral:7b"
            ),
            inputs=[],
            outputs=[placeholder]
        )
        demo.launch(debug=True, share=True)

    return demo

if __name__ == "__main__":
    # start_ollama()
    ui()