import gradio as gr
import utils



# Araclip demo 
with gr.Blocks() as demo_araclip:

    gr.Markdown("## Choose the dataset")

    dadtaset_select = gr.Radio(["XTD dataset", "Flicker 8k dataset"], value="XTD dataset", label="Dataset", info="Which dataset you would like to search in?")

    gr.Markdown("## Input parameters")
    
    txt = gr.Textbox(label="Text Query (Caption)")
    num = gr.Slider(label="Number of retrieved image", value=1, minimum=1)
    

    with gr.Row():
        btn = gr.Button("Retrieve images", scale=1)

    gr.Markdown("## Retrieved Images")

    gallery = gr.Gallery(
        label="Generated images", show_label=True, elem_id="gallery"
    , columns=[5], rows=[1], object_fit="contain", height="auto")


    with gr.Row():
        lables = gr.Label(label="Text image similarity") 

    with gr.Row():
        
        with gr.Column(scale=1):
            gr.Markdown("<div style='text-align: center; font-size: 24px; font-weight: bold;'>Data Retrieved based on Images Similarity</div>")

            json_output = gr.JSON()

        with gr.Column(scale=1):
            gr.Markdown("<div style='text-align: center; font-size: 24px; font-weight: bold;'>Data Retrieved based on Text similarity</div>")
            json_text = gr.JSON()


    btn.click(utils.predict, inputs=[txt, num, dadtaset_select], outputs=[gallery,lables, json_output, json_text])


    gr.Examples(
        examples=[["تخطي لاعب فريق بيتسبرج بايرتس منطقة اللوحة الرئيسية في مباراة بدوري البيسبول", 5], 
                  ["وقوف قطة بمخالبها على فأرة حاسوب على المكتب", 10],
                  ["صحن به شوربة صينية بالخضار، وإلى جانبه بطاطس مقلية وزجاجة ماء", 7]],
        inputs=[txt, num, dadtaset_select],
        outputs=[gallery,lables, json_output, json_text],
        fn=utils.predict,
        cache_examples=False,
    )

# mclip demo 
with gr.Blocks() as demo_mclip:

    gr.Markdown("## Choose the dataset")

    dadtaset_select = gr.Radio(["XTD dataset", "Flicker 8k dataset"], value="XTD dataset", label="Dataset", info="Which dataset you would like to search in?")


    gr.Markdown("## Input parameters")
    
    txt = gr.Textbox(label="Text Query (Caption)")
    num = gr.Slider(label="Number of retrieved image", value=1, minimum=1)

    with gr.Row():
        btn = gr.Button("Retrieve images", scale=1)

    gr.Markdown("## Retrieved Images")

    gallery = gr.Gallery(
        label="Generated images", show_label=True, elem_id="gallery_mclip"
    , columns=[5], rows=[1], object_fit="contain", height="auto")

    
    lables = gr.Label() 

    with gr.Row():
        
        with gr.Column(scale=1):
            gr.Markdown("## Images Retrieved")
            json_output = gr.JSON()

        with gr.Column(scale=1):
            gr.Markdown("## Text Retrieved")
            json_text = gr.JSON()

    btn.click(utils.predict_mclip, inputs=[txt, num, dadtaset_select], outputs=[gallery,lables, json_output, json_text])

    gr.Examples(
        examples=[["تخطي لاعب فريق بيتسبرج بايرتس منطقة اللوحة الرئيسية في مباراة بدوري البيسبول", 5], 
                  ["وقوف قطة بمخالبها على فأرة حاسوب على المكتب", 10],
                  ["صحن به شوربة صينية بالخضار، وإلى جانبه بطاطس مقلية وزجاجة ماء", 7]],
        inputs=[txt, num, dadtaset_select],
        outputs=[gallery,lables, json_output, json_text],
        fn=utils.predict_mclip,
        cache_examples=False,
    )


# Group the demos in a TabbedInterface 
with gr.Blocks() as demo:

    gr.Markdown("<font color=red size=10><center>AraClip: Arabic Image Retrieval Application</center></font>")

    gr.TabbedInterface([demo_araclip, demo_mclip], ["Our Model", "Mclip model"])


if __name__ == "__main__":
    
    demo.launch()