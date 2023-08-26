# Autoblur

An open-source solution for blurring sensible objects in a image

## <div align="center">Documentation</div>

See the [docs](URL) for full documentation.

## <div align="center">Instalation</div>

#### Client install
```bash
git clone https://github.com/pydoni/autoblur  # clone
cd autoblur
pip install -r requirements.txt  # install
```

#### Api
```bash
pip install autoblur  # install
```

## <div align="center">Quick-start</div>

#### Api
```bash
import autoblur
ab_core= autoblur.ab_core() #Load models
img = ab_core.apply_blur(img_to_blur) # Receives a np.ndarray() img and returns a image with license plates blurred
```

#### Using blur.py
```bash
python3 blur.py --params
```
## <div align="center">Compatible models</div>
If you want to blur objects that are not available in this tool, you can load your own weights and filter the classes following this tutorial in the docs.
Currently the tool supports only yolov5 and yolov8 architecture.

## <div align="center">List of open-source weights</div>
Through here you can find some weights for different objects and hardware, you can use these weights by defining their names and model on the api/blur.py params

## <div align="center">Contribute</div>
## <div align="center">Contact</div>

