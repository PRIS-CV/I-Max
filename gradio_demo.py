import gradio as gr
from gradio_imageslider import ImageSlider
import torch
from pipeline_flux_imax import FluxPipeline
from transformer_flux import FluxTransformer2DModel
import os
import json

bfl_repo="black-forest-labs/FLUX.1-dev"

# Initialize the model
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, torch_dtype=torch.bfloat16)
pipe.transformer = transformer
pipe.scheduler.config.use_dynamic_shifting = False
pipe.to("cuda")

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def generate_image(prompt, image_size, seed, ntk_factor, num_inference_steps1, num_inference_steps2, guidance_scale1, guidance_scale2, time_shift_1, time_shift_2, dwt_level, guidance_schedule):
    height, width = map(int, image_size.split('x'))
    torch.random.manual_seed(int(seed))
    print(type(dwt_level), type(time_shift_2))
    images = pipe(prompt=prompt,
                  num_inference_steps1=num_inference_steps1, num_inference_steps2=num_inference_steps2,
                  guidance_scale1=guidance_scale1, guidance_scale2=guidance_scale2,
                  height=height, width=width,
                  ntk_factor=ntk_factor,
                  return_dict=False,
                  time_shift_1=time_shift_1, time_shift_2=time_shift_2,
                  dwt_level=int(dwt_level),
                  proportional_attention=True,
                  text_duplication=True,
                  swin_pachify=True,
                  guidance_schedule=guidance_schedule,
                  )

    # Save images and parameters
    image_1, image_2 = images[1], images[0]
    index = len([name for name in os.listdir('results') if os.path.isfile(os.path.join('results', name)) and name.endswith('.jpeg')]) // 2 + 1
    image_1.save(f'results/{index}_native.jpeg', 'JPEG')
    image_2.save(f'results/{index}_extrapolated.jpeg', 'JPEG')

    params = {
        'prompt': prompt,
        'image_size': image_size,
        'seed': seed,
        'ntk_factor': ntk_factor,
        'num_inference_steps1': num_inference_steps1,
        'num_inference_steps2': num_inference_steps2,
        'guidance_scale1': guidance_scale1,
        'guidance_scale2': guidance_scale2,
        'time_shift_1': time_shift_1,
        'time_shift_2': time_shift_2,
        'guidance_schedule': guidance_schedule,
        'dwt_level': dwt_level
    }

    with open(f'results/{index}.json', 'w') as f:
        json.dump(params, f, indent=4)

    return (image_1, image_2)

# Gradio Interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", value="Create a bold and dynamic text design for the word \"I-MAX\" with each letter filled with vibrant and high-fashion photography scenes. Incorporate a mix of models striking elegant poses, cameras flashing, and creative studio setups. Highlight the diversity of the modeling world with a variety of model expressions, runway moments, and behind-the-scenes shots. Use sleek, modern colors that reflect professionalism, creativity, and innovation, integrating camera lenses, softboxes, and fashion accessories within the letters to emphasize the photography and modeling theme. The overall design should feel high-end, artistic, and tailored for a professional audience."),
        gr.Dropdown(label="Image Size", choices=["2048x2048", "3072x3072", "4096x4096"], value="4096x4096"),
        gr.Number(label="Seed", value=25),
        gr.Slider(label="NTK Factor", minimum=1, maximum=15, step=1, value=10),
        gr.Slider(label="Num Inference Steps 1", minimum=1, maximum=100, step=1, value=30),
        gr.Slider(label="Num Inference Steps 2", minimum=1, maximum=100, step=1, value=20),
        gr.Slider(label="Guidance Scale 1", minimum=1, maximum=15, step=0.1, value=3.5),
        gr.Slider(label="Guidance Scale 2", minimum=1, maximum=15, step=0.1, value=5),
        gr.Slider(label="Time Shift 1", minimum=1, maximum=15, step=1, value=3),
        gr.Slider(label="Time Shift 2", minimum=1, maximum=15, step=1, value=10),
        gr.Slider(label="DWT Level", minimum=1, maximum=5, step=1, value=1),
        gr.Dropdown(label="Guidance Schedule", choices=["cosine_decay", "cosine_shift", "constant", "disable"], value="cosine_decay")
    ], 
    outputs=ImageSlider(label="Illustration of the native-resolution guidance and the final result.", type="pil", slider_color="pink"),
    # outputs=gr.Gallery(label="Illustration of the native-resolution guidance and the final result."),
    title="I-Max Gradio Demo",
    description="Generate images with the I-Max + Flux.1-dev Pipeline."
)

demo.launch(share=True, server_port=7860, server_name='0.0.0.0')
