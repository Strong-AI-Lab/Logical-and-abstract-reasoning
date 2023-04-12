
class BERTConfig:
    # dataset
    data_dir = "../"  # Subject for data path: Biology, Law
    train_data_file_name = "Synthetic_xfm_t5wtense_logical_equivalence_train.csv"
    validate_data_file_name = "Synthetic_xfm_t5wtense_logical_equivalence_validation.csv"
    test_data_file_name = "synthetic_logical_equivalence_sentence_pair_testset.csv"

    # pretrained model
    pretrained_model_name = "./Transformers/bert-base-cased/"  # pretrained model: BERT, BioBERT, RoBERTa, SBERT

    # save model
    saved_fig_dir = "./result/figure/"
    saved_model_dir = "./result/saved_models/"  # save model after fine-tune

    # load model
    load_model_sub_dir_name = "epoch_3"  # for load model from a specific sub dir, e.g: epoch_5
    predict_result_dir = "./result/predict/"

    # train + predict parameters
    GPU_ID = "1"
    num_labels = 2  # The number of output labels -- 1 for MSE Loss Regression; other for classification.
    batch_size = 16  # for DataLoader (when fine-tuning BERT on a specific task, 16 or 32 is recommended)
    epochs = 4  # Number of training epochs (we recommend between 2 and 4)
    lr = 5e-5  # Optimizer parameters: learning_rate - default is 5e-5, our notebook had 2e-5
    eps = 1e-8  # Optimizer parameters: adam_epsilon  - default is 1e-8.
    seed = 2022  # Set the seed value all over the place to make this reproducible.

    pct_close = 0.1  # predict correct threshold


def parse(self, kwargs):
    '''
    user can update the default hyperparamter
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            raise Exception('config has No key: {}'.format(k))
        setattr(self, k, v)

    print('*************************************************')
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print("{} => {}".format(k, getattr(self, k)))

    print('*************************************************')


BERTConfig.parse = parse
opt = BERTConfig()
