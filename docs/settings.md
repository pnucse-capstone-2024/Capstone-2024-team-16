## Server Setting
```sh
docker pull pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
```

```sh
# create
docker run -dit --name aicasso -v ./Barbershop:/barbershop {docker id}
docker exec -it aicasso bash

# install lib
apt-get update
apt-get install git
cd /{workdir}
pip install -r requirements.txt

# commit
docker commit aicasso aicasso
```

## Model

yolov8n-face

.pt to .onnx 파일 변환 : model/pt2onnx.py

> 생각보다 라이브러리들이 많이 필요한 것 같음 warning 많이 뜸
> 변환시 omegaconf, onnx, pytorch
> 추론할 때 onnxruntinme필요
>
