from GenerateFeatures import extract_featutres
from Train import k_fold_cv
from Train import join_train_test
from Train import split_train

if __name__ == "__main__":
    labelling_criteria = 'Death'
    outdir = 'Labelled by ' + labelling_criteria

    train_filename = 'Train_labelled_by_' + labelling_criteria
    test_filename = 'Test_labelled_by_' + labelling_criteria

    # extract_featutres(train_filename, outdir)
    # extract_featutres(test_filename, outdir)

    split_train(train_filename, test_filename, outdir)

    data_x, data_y = join_train_test(train_filename, test_filename, outdir)
    k_fold_cv(data_x, data_y, outdir)

