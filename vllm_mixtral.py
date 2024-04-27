import os
from modal import Image, Stub, gpu, method, asgi_app
from fastapi import FastAPI

MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#BASE_MODEL = "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ"

GPU_CONFIG = gpu.A100(memory=80, count=2)


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns="*.pt",  # Using safetensors
    )
    move_cache()


vllm_image = (
    Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
    ).pip_install(
        "pytesseract",
        "vllm",
        "huggingface_hub",
        "hf-transfer",
        "torch",
        "pillow==10.2.0",
        "fastapi",
        #force_build=True
    ).env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    ).apt_install(
        "tesseract-ocr",
        "tesseract-ocr-deu"
    ) #"libjpeg,", "libjpeg-dev", "libfreetype6", "libfreetype6-dev", "zlib1g-dev"
    .run_function(download_model_to_folder, timeout=60 * 20)
)

with vllm_image.imports():
    from vllm import LLM
    import subprocess
    from vllm import SamplingParams
    import io
    from PIL import Image
    import pytesseract


stub = Stub("mixtral")


@stub.cls(
    gpu=GPU_CONFIG,
    timeout=600,
    container_idle_timeout=10,
    image=vllm_image,
)
class Mixtral:
    def __enter__(self):
        if GPU_CONFIG.count > 1:
            # Wegen https://github.com/vllm-project/vllm/issues/1116
            import ray
            ray.shutdown()
            ray.init(num_gpus=GPU_CONFIG.count)
        self.engine = LLM(
            model=MODEL_DIR,
            #quantization="awq",
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
        )

        self.template = "<s> [INST] {user} [/INST] "

        # Performance improvement from https://github.com/vllm-project/vllm/issues/2073#issuecomment-1853422529
        if GPU_CONFIG.count > 1:
            RAY_CORE_PIN_OVERRIDE = "cpuid=0 ; for pid in $(ps xo '%p %c' | grep ray:: | awk '{print $1;}') ; do taskset -cp $cpuid $pid ; cpuid=$(($cpuid + 1)) ; done"
            subprocess.call(RAY_CORE_PIN_OVERRIDE, shell=True)

    def run_inference(self, user_question):

        sampling_params = SamplingParams(
            temperature=0.75,
            max_tokens=1024,
            repetition_penalty=1.1,
        )

        instruction = "Konvertiere einen unstrukturierten, schlecht formatierten Rezepttext in das standardisierte schema.org \
Rezept-Format. Dieser Text ist das Ergebnis von einem OCR Programm, daher könnten Sonderzeichen die im Deutschen Alphabet nicht vorkommen erscheinen. \
Extrahiere relevante Informationen wie Zutaten, Kochschritte, Kochzeit und Portionengröße und füge sie in das JSON-Objekt im schema.org \
Rezept-Format ein. Nutze alle Informationen die verwertbar sind für das Schema. Sollte nicht klar sein zu welcher Kategorie ein Rezept \
gehört sollst du eine nutzen die Sinn ergibt. Stelle sicher, dass die Lösung sinnvoll ist. Achte darauf dass keine Informationen \
erfunden werden. Deine Antwort soll ein schema.org recipe JSON sein und niemals etwas anderes."

        merged = f"{instruction} \n \n {user_question}"

        result = self.engine.generate(
            self.template.format(user=merged),
            sampling_params
        )

        for output in result:
            generated_text = output.outputs[0].text

        return generated_text

    @method()
    def run_image_inference(self, image):
        tmp = io.BytesIO(image)
        pil_img = Image.open(tmp)
        ocr_img = pytesseract.image_to_string(pil_img, lang='deu')
        response = self.run_inference(ocr_img)
        return response



web_app = FastAPI()

@stub.function()
@asgi_app(label="mixtral")
def app():
    #import fastapi
    from fastapi import File, UploadFile
    from fastapi.responses import JSONResponse

    @web_app.get("/completion/{question}")
    async def completion(question):
        # API get test
        question = "Render following in proper German and return everything in a valid JSON and nothing else: \n \nPro Portion\n\n    630 kcal\n    43 g EiweiÃŸ\n    38 g Fett\n    19 g Kohlenhydrate"

        model = Mixtral()
        res = model.run_inference.remote(question)

        return JSONResponse(content=res)

    @web_app.post("/uploadimage/")
    async def create_upload_file(file: UploadFile = File(...)):
        image = await file.read()
        model = Mixtral()
        res = model.run_image_inference.remote(image)

        return JSONResponse(content=res)

    return web_app
