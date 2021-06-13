import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('data/multiclass.csv')
    X = df['id']
    y = df['attribute_ids']
    labels = list(set(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)

    train_df = pd.DataFrame({'id': X_train, 'attribute_ids': y_train})
    test_df = pd.DataFrame({'id': X_test, 'attribute_ids': y_test})

    print('Original:')
    print(df['attribute_ids'].value_counts(normalize=True)[:5])
    print('Train:')
    print(train_df['attribute_ids'].value_counts(normalize=True)[:5])
    print('Test:')
    print(test_df['attribute_ids'].value_counts(normalize=True)[:5])

    train_df.to_csv('data/multiclass_train.csv')
    test_df.to_csv('data/multiclass_test.csv')
