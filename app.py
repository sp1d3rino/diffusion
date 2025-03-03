import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "foto di un haker in tuta nera che sta scrivendo del codice seduto all scrivania con 6 monitor davanti. Nella stanza la luce e' scarsa ma ci sono i led a illuminare tutto. il soggetto e' di spalle"
image = pipe(prompt).images[0]
    
image.save("image4.png")

