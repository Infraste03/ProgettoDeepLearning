Nella seguente Repository sono presenti le seguenti cartelle:
1) CGAN
2) WGAN
3) Pesi
   
#1 CGAN
All interno della cartella è possibile visionare i file necessari per poter eseguire la CGAN tramite il segunete comando : 
python ./GANBCE.py [train | test]
In tale file sono importati i file del Dataloader(StarGan) ,Generatore, Discriminatore, Test e il file CLASSIFICATION_AND_ADV.py nel quale viene combinata l' adversarial loss con la classification loss
./CLASSIFICATION_AND_ADV.py [train | test]

#WGAN

All interno della cartella è possibile visionare i file necessari per poter eseguire la WGAN tramite i seguneti comandi : 
python ./WGANClipping.py [train | test] per eseguire il file nel quale è implementato il codice con il clipping dei pesi
python ./WGANGradient.py [train | test] per eseguire il file nel quale è implemtata la gradient penalty
In tale file sono importati i file del Dataloader(StarGan) ,Generatore, Discriminatore, Test

#Pesi
sono riportati alcuni dei pesi derivanti dal training 
