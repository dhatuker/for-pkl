from Lib.LDA import LDA, LDA_Proses

def main():
    data = LDA()
    lda = data.run()
    del data

if __name__ == '__main__':
    main()
