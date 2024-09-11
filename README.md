Nella seguente repository sono presenti:
- CGAN (Conditional GAN) 

  Nella cartella CGAN è possibile trovare i file necessari per eseguire la CGAN.
  
  Esecuzione
  #
  Per avviare l'addestramento o il test, si può utilizzare  il comando seguente:
  
  python ./GANBCE.py [train | test]
  Descrizione dei file
  Dataloader (StarGan): per la gestione del caricamento dei dati.
  
  Generatore: implementazione del modello generatore.
  
  Discriminatore: implementazione del modello discriminatore.
  
  Test: modulo per eseguire i test.
  
  CLASSIFICATION_AND_ADV.py: script che combina l'adversarial loss con la classification loss. 
  Per eseguire:
  
  python ./CLASSIFICATION_AND_ADV.py [train | test]
2. WGAN (Wasserstein GAN)
Nella cartella WGAN sono inclusi i file per eseguire le versioni della WGAN con clipping dei pesi e gradient penalty.

Esecuzione
Per eseguire la WGAN con clipping dei pesi, utilizza il comando:

bash
Copia codice
python ./WGANClipping.py [train | test]
Per eseguire la WGAN con gradient penalty, utilizza il comando:

bash
Copia codice
python ./WGANGradient.py [train | test]
Descrizione dei file
Dataloader (StarGan): per la gestione del caricamento dei dati.
Generatore: implementazione del modello generatore.
Discriminatore: implementazione del modello discriminatore.
Test: modulo per eseguire i test.
3. Pesi
La cartella Pesi contiene alcuni dei pesi ottenuti durante il training dei modelli.
