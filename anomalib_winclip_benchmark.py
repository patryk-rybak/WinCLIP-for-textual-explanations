from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models.image import WinClip

datamodule = MVTecAD(root="./dataset", category='screw')
model = WinClip()

engine = Engine()
engine.test(model=model, datamodule=datamodule)
