# Arabic tweets Sentiment Analysis using CNN-LSTM model and Retrofitting

The document for the project can be found [here](https://1drv.ms/w/s!AsU-pnqVXBB3gfNE_TW9QsKlAxkMxA)
### running the model steps
1. Generating a space separated word embeddings
  - copy the dataset from [here](https://1drv.ms/u/s!AsU-pnqVXBB3gfMcN6C6lDaXE-GfDw) to the tweets_sg_100 folder
  - Run the "python Parse.py" script to generate it
  - the output is saved to AraWordVec.txt file
2. Create the semantic lexicon file
  - Go to LexiconCreator folder
  - run the "python LexCreate.py" file to generate the AraSynonyms.txt file from awn.xml
3. Create the retrofitted word embeddings
  - go to retrofitting folder
  - run "python retrofit.py -i ..\tweets_sg_100\AraWordVec.txt -l ..\LexiconCreator\AraSynonyms.txt -o RetrofitAraWordVec.txt"
  - output is saved to RetrofitAraWordVec.txt
 4. Run the model
  - go to CS291K folder
  - Training the model
     - make sure original word vectors file and retrofitted file are in that folder, you can also download from [here](https://1drv.ms/u/s!AsU-pnqVXBB3gfMdtCpBl4fdcwsroA)
     - Train on original vectors using the command "python train.py -e AraWordVec.txt -d 100 -p 10 -o runs"
     - Train on retrofitted vectors using the command "python train.py -e RetrofitAraWordVec.txt -d 100 -p 10 -o runs"
     - use "python train.py -h" for usage help
   - Testing the model
    - use the command "python test.py -e <word embeddings file> -m <Checkpoints folder inside runs folder> -i <latest checkpoint .meta file name> -d 100"
    - use "python test.py -h" for usage help
  
