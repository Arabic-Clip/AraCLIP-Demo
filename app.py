import gradio as gr
import utils



# Araclip demo 
with gr.Blocks() as demo_araclip:

    gr.Markdown("## Choose the dataset")

    dadtaset_select = gr.Radio(["XTD dataset", "Flicker 8k dataset"], value="XTD dataset", label="Dataset", info="Which dataset you would like to search in?")

    gr.Markdown("## Input parameters")
    
    txt = gr.Textbox(label="Text Query")
    num = gr.Slider(label="Number of retrieved image", value=1, minimum=1, step=1)
    

    with gr.Row():
        btn = gr.Button("Retrieve images", scale=1)

    gr.Markdown("## Retrieved Images")

    gallery = gr.Gallery(
        show_label=False, elem_id="gallery"
    , columns=[5], rows=[1], object_fit="contain", height="auto")


    with gr.Row():
        lables = gr.Label(label="Text-image similarity") 

 
    btn.click(utils.predict, inputs=[txt, num, dadtaset_select], outputs=[gallery,lables])


    gr.Examples(
        examples=[["تخطي لاعب فريق بيتسبرج بايرتس منطقة اللوحة الرئيسية في مباراة بدوري البيسبول", 5], 
                  ["وقوف قطة بمخالبها على فأرة حاسوب على المكتب", 10],
                  ["صحن به شوربة صينية بالخضار، وإلى جانبه بطاطس مقلية وزجاجة ماء", 7]],
        inputs=[txt, num, dadtaset_select],
        outputs=[gallery,lables],
        fn=utils.predict,
        cache_examples=False,
    )

# mclip demo 
with gr.Blocks() as demo_mclip:

    gr.Markdown("## Choose the dataset")

    dadtaset_select = gr.Radio(["XTD dataset", "Flicker 8k dataset"], value="XTD dataset", label="Dataset", info="Which dataset you would like to search in?")


    gr.Markdown("## Input parameters")
    
    txt = gr.Textbox(label="Text Query")
    num = gr.Slider(label="Number of retrieved image", value=1, minimum=1, step=1)

    with gr.Row():
        btn = gr.Button("Retrieve images", scale=1)

    gr.Markdown("## Retrieved Images")

    gallery = gr.Gallery(
        label="Generated images", show_label=True, elem_id="gallery_mclip"
    , columns=[5], rows=[1], object_fit="contain", height="auto")

    
    lables = gr.Label() 

    btn.click(utils.predict_mclip, inputs=[txt, num, dadtaset_select], outputs=[gallery,lables])

    gr.Examples(
        examples=[["تخطي لاعب فريق بيتسبرج بايرتس منطقة اللوحة الرئيسية في مباراة بدوري البيسبول", 5], 
                  ["وقوف قطة بمخالبها على فأرة حاسوب على المكتب", 10],
                  ["صحن به شوربة صينية بالخضار، وإلى جانبه بطاطس مقلية وزجاجة ماء", 7]],
        inputs=[txt, num, dadtaset_select],
        outputs=[gallery,lables],
        fn=utils.predict_mclip,
        cache_examples=False,
    )


# Define custom CSS to increase the size of the tabs
custom_css = """
.gr-tabbed-interface .gr-tab {
    font-size: 50px;  /* Increase the font size */
    padding: 10px;    /* Increase the padding */
}
"""

# Group the demos in a TabbedInterface 
with gr.Blocks(analytics_enabled=False) as demo:

    # gr.Image("statics/logo_araclip.png")
    gr.Markdown("""
            <center> <img src="/file=statics/logo_araclip.png" alt="Imgur" style="width:200px"></center>
                """)
    gr.Markdown("<center> <font color=red size=10>AraClip: Arabic Image Retrieval Application</font></center>")

    gr.Markdown("""
            <font size=4>   To run the demo 🤗, please select the model, then the dataset you would like to search in, enter a text query, and specify the number of retrieved images.</font>
                    
                """)



    gr.TabbedInterface([demo_araclip, demo_mclip], ["Our Model", "Mclip model"], css=custom_css)

    gr.Markdown(
        """
            If you find this work helpful, please help us to ⭐ the repositories in <a href='https://github.com/Arabic-Clip' target='_blank'>Github Organization</a>. Thank you! 
            
            ---
            📝 **Citation**
            
            To be shared soon.
            
            📋 **License**
            """
            )
if __name__ == "__main__":
    
    demo.launch(  server_name="0.0.0.0", share=True, allowed_paths=["statics",'./', '.'])