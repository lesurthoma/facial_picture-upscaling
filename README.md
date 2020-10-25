## requirement
- pip3
- virtualenv

## installation
let's create a virtualenv : 
```bash
python3 -m virtualenv venv
```
then activate the virtualenv
```bash
source venv/bin/activate
```
and then install requirements of the project
```bash
pip install -r requirements.txt
```

## activate virtualenv
```bash
source venv/bin/activate
```

## deactivate virtualenv
```bash
deactivate
```

## train the model
```bash
python src/train.py
```

## use the model
```bash
python src/upscale_face.py SOURCE_IMAGE_PATH DESTINATION_FILE_PATH
```