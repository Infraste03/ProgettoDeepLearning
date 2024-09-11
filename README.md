
# Deep Learning and Generative Models  
**Project Assignment #10**

## Project Objective  
L'obiettivo del progetto è creare una **cGAN** in grado di generare immagini che appartengono a una specifica classe, ad esempio un volto umano con capelli biondi o un sorriso.

### Dataset  
- **CelebA dataset**: contiene immagini di volti umani con annotazioni degli attributi (es. colore dei capelli, presenza di occhiali, ecc.).
  - È scaricare il dataset e le relative annotazioni da [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
  - In alternativa, si può usare lo script `download.sh` dalla repository [StarGAN](https://github.com/yunjey/stargan).

### Network Model  
- **Conditional GAN (cGAN)**: usato per generare immagini appartenenti a specifiche classi.
  - Esperimenti suggeriti: combinare **adversarial loss** con **Wasserstein loss**.
  
### Dettagli Implementativi  
- Il **dataloader** deve restituire un'immagine e le sue corrispondenti etichette di attributo (si può utilizzare il **StarGAN Dataloader** come punto di partenza da [qui](https://github.com/yunjey/stargan)).
- Si può combinare la **adversarial loss** con una **classification loss** per migliorare i risultati (opzionale).
- È possibile decidere a priori il numero di attributi su cui addestrare il modello. Ad esempio, puoi selezionare 5 attributi e sperimentare su quelli.

---

## Repository Structure

Questa repository contiene implementazioni di modelli GAN, inclusi **CGAN** e **WGAN**, con alcuni dei pesi ottenuti durante il training.

### Contenuto delle cartelle:
1. **CGAN**
2. **WGAN**
3. **Pesi**

---

## 1. CGAN (Conditional GAN)

Nella cartella `CGAN` sono presenti i file necessari per eseguire la CGAN.

### Esecuzione:
Per avviare l'addestramento o il test, è possibile utlizzare il seguente comando:

```bash
python ./GANBCE.py [train | test]
```

### Descrizione dei file:
- **Dataloader (StarGan)**: gestione del caricamento dei dati.
- **Generatore**: implementazione del modello generatore.
- **Discriminatore**: implementazione del modello discriminatore.
- **Test**: modulo per eseguire i test.
- **CLASSIFICATION_AND_ADV.py**: script che combina l'adversarial loss con la classification loss.
- Per eseguire:

```bash
python ./CLASSIFICATION_AND_ADV.py [train | test]
```

---

## 2. WGAN (Wasserstein GAN)

Nella cartella `WGAN` sono inclusi i file per eseguire le versioni della WGAN con **clipping dei pesi** e **gradient penalty**.

### Esecuzione:
- Per eseguire la WGAN con **clipping dei pesi**, è possibile utlizzare il comando:

  ```bash
  python ./WGANClipping.py [train | test]
  ```

- Per eseguire la WGAN con **gradient penalty**, è possibile utlizzare il comando:

  ```bash
  python ./WGANGradient.py [train | test]
  ```

### Descrizione dei file:
- **Dataloader (StarGan)**: gestione del caricamento dei dati.
- **Generatore**: implementazione del modello generatore.
- **Discriminatore**: implementazione del modello discriminatore.
- **Test**: modulo per eseguire i test.

---

## 3. Pesi

La cartella `Pesi` contiene alcuni dei pesi ottenuti durante il training dei modelli.

## Esempio immagini generate
![image](https://github.com/user-attachments/assets/2e573def-2793-411e-880f-920a7d59f9b9)




