Data Source: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download-directory&select=HAM10000_metadata.csv

Model History:
CNN1: baseline_cnn_0707_161948

ResNetFrozen1: ResNetFrozen_0707_204931
    - 50 epochs
ResNetFrozen2: ResNetFrozen_0707_212302
    - 100 epochs


ResNetFineTuned1: ResNetFineTuned-Layer4Unfrozen_0708_010323
    - Unfrozen Layers: ['layer4', 'fc']


Future plans: 
 - Tensorboard inside the JupyterNotebooks?
 - ResNetFineTuned with heavier melenoma weights?
 - ResNetFineTuned with 3 and 4 unfrozen.
 - README explanation
 - Presentation added to GitHub
