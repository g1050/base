from transformers import pipeline
from transformers import AutoModel
def ppl():
    pipe = pipeline("text-classification")
    result = pipe("This restaurant is awesome")
    print(result)
def vgg():
    # Load model directly
    model = AutoModel.from_pretrained("glasses/vgg11_bn")
    print(model)
    
if __name__=="__main__":
    vgg()