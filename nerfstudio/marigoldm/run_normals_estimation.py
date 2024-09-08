import diffusers
if __name__ == "__main__":
    model_path = "D:/ML_AI_DL_Projects/projects_repo/Marigold/prs-eth/marigold-normals-v0-1"

    model_paper_kwargs = {
        diffusers.schedulers.DDIMScheduler: {
            "num_inference_steps": 10,
            "ensemble_size": 10,
        },
        diffusers.schedulers.LCMScheduler: {
            "num_inference_steps": 4,
            "ensemble_size": 5,
        },
    }

    image = diffusers.utils.load_image("D:/ML_AI_DL_Projects/projects_repo/RDV/nerfstudio/marigold/input/in-the-wild_example/example_2.jpg")

    pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(model_path).to("cuda")
    pipe_kwargs = model_paper_kwargs[type(pipe.scheduler)]

    depth = pipe(image, **pipe_kwargs)

    vis = pipe.image_processor.visualize_normals(depth.prediction)
    vis[0].save("D:/ML_AI_DL_Projects/projects_repo/RDV/nerfstudio/marigold/output/example_2.png")