import pandas as pd
import numpy as np
import torch
import evaluate
from tqdm import tqdm
from textwrap import TextWrapper
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import pandas_datareader
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
from torch.nn.functional import softmax


PRICE_CHANGE_THRESHOLD= 0.1  # Define threshold for significant price change
SENTIMENT_THRESHOLD = 0.8  # Define threshold for strong sentiment

def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_evaluation_df_and_plot(true_labels, pred_labels, target_names, model_name):
    """
    Function to evaluate a model's predictions by calculating accuracy, F1 score, confusion matrix and then visualizing the results.

    This function takes as input the true and predicted labels, the target names, and the model name. It calculates
    the confusion matrix and classification report (which includes precision, recall, and F1 score for each class),
    and displays them in a neatly formatted dataframe. It also creates a visual representation of the confusion matrix.

    Parameters
    ----------
    true_labels : array-like
        The true labels of the dataset.
    pred_labels : array-like
        The predicted labels as returned by the model.
    target_names : list
        A list of strings representing the names of the classes.
    model_name : str
        The name of the model being evaluated.

    Returns
    -------
    evaluation_df : pandas DataFrame
        A DataFrame containing the accuracy and F1 score of the model, as well as the F1 score for each individual class.

    Displays
    --------
    - A DataFrame containing the model's accuracy, overall F1 score, and per-class F1 scores.
    - A classification report.
    - A heatmap of the confusion matrix.

    Raises
    ------
    ValueError
        If the lengths of 'true_labels' and 'pred_labels' are not equal.
    """

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=target_names, output_dict=True)

    # Create a dictionary to store the evaluation metrics
    evaluation_results = {
        "Model": [model_name],
        "Accuracy": [accuracy_score(true_labels, pred_labels)],
        "F1 Score": [f1_score(true_labels, pred_labels, average='weighted')],
    }

    # Add F1 scores for each class to the dictionary
    for idx, name in enumerate(target_names):
        evaluation_results[name + " F1"] = [report[name]["f1-score"]]

    # Convert the dictionary to a DataFrame
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.set_index("Model", inplace=True)

    # Print the evaluation DataFrame and classification report
    print(evaluation_df)
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=target_names))

    # Set up the figure for plotting
    fig, ax = plt.subplots()

    # Plot the confusion matrix
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{model_name}\nAccuracy: {evaluation_results['Accuracy'][0]:.2f}\nF1 Score: {evaluation_results['F1 Score'][0]:.2f}")
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    # Add x and y axis labels
    ax.set_xlabel("Predicted Class", fontsize=18)
    ax.set_ylabel("True Class", fontsize=18)

    # Display F1 scores in the confusion matrix
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            text_color = "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
            ax.text(j, i, f"{conf_matrix[i, j]}\n({report[target_names[i]]['f1-score']:.2f})",
                    ha="center", va="center", color=text_color, fontsize=12)

    plt.show()

    return evaluation_df


def evaluate_model(model, tokenizer, test_set, target_names, model_name="", is_dataframe=True):
    # Prepare the data
    if is_dataframe:
        test_true_labels = test_set['label'].tolist()
        test_texts = test_set['sentence'].tolist()
    else:
        test_true_labels = test_set['label']
        test_texts = test_set['text']

    # Tokenize the test data
    test_inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)

    # Get predictions
    with torch.no_grad():
        logits = model(**test_inputs).logits
    test_pred_labels = np.argmax(logits.numpy(), axis=1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_true_labels, test_pred_labels)
    f1 = f1_score(test_true_labels, test_pred_labels, average='weighted')
    conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
    report = classification_report(test_true_labels, test_pred_labels, target_names=target_names, output_dict=True, digits=4)

    # Create a dictionary to store the evaluation metrics
    evaluation_results = {
        "Model": [model_name],
        "Accuracy": [accuracy],
        "F1 Score": [f1],
    }

    # Add F1 scores for each class to the dictionary
    for idx, name in enumerate(target_names):
        evaluation_results[name + " F1"] = [report[name]["f1-score"]]

    # Convert the dictionary to a DataFrame
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.set_index("Model", inplace=True)

    # Print the evaluation DataFrame and classification report
    print(evaluation_df)
    print("Classification Report:")
    print(classification_report(test_true_labels, test_pred_labels, target_names=target_names, digits=4))

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{model_name}\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}")
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    # Add x and y axis labels
    ax.set_xlabel("Predicted Class", fontsize=16)
    ax.set_ylabel("True Class", fontsize=16)

    # Display F1 scores in the confusion matrix
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            text_color = "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
            ax.text(j, i, f"{conf_matrix[i, j]}\n({report[target_names[i]]['f1-score']:.4f})",
                    ha="center", va="center", color=text_color, fontsize=12)

    plt.show()

    return evaluation_df


class TextClassifier:
    """
    A class to evaluate multiple models for text classification tasks.

    Attributes
    ----------
    test_set : DataFrame
        The test dataset which is used for evaluating the models. It should include 'text' and 'label' columns.
    models : dict
        A dictionary where keys are model names and values are the model checkpoint paths.
    target_names : list
        A list of class names corresponding to the labels in the 'label' column of the test dataset.
    num_examples : int, optional
        The number of examples to use from the test set for evaluation. Default is 100.
    seed_value : int, optional
        The seed value for random operations in numpy and torch. Default is 0.
    label_mapping : dict, optional
        A dictionary mapping original labels to new ones. If not provided, default is a mapping from range(len(target_names)) to itself.

    Methods
    -------
    set_seed(seed_value=42)
        Sets the seed for numpy, torch, and cudnn to ensure results are reproducible.
    evaluate_models(num_columns=3, figsize=(15, 10))
        Evaluates all models on the test set and provides a summary of the results.
        This includes accuracy, F1 score, and a confusion matrix for each model.
        The results are also plotted for a visual comparison between models.
    """

    def __init__(self, test_set, models, target_names, num_examples=100, seed_value=0, label_mapping=None):
        self.set_seed(seed_value)
        self.test_set = test_set
        self.models = models
        self.target_names = target_names
        self.num_examples = num_examples
        self.label_mapping = label_mapping or {i: i for i in range(len(target_names))}

    @staticmethod
    def set_seed(seed_value=42):
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def evaluate_models(self, num_columns=3, figsize=(15, 10)):
        evaluation_results = []

        num_models = len(self.models)
        num_rows = (num_models + num_columns - 1) // num_columns

        fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize, sharey=True)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        check = u'\u2705'

        for idx, (model_name, model_checkpoint) in enumerate(self.models.items()):
            print(f"{check} Evaluating {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(self.target_names))

            test_texts = self.test_set["text"][:self.num_examples]
            test_true_labels = [self.label_mapping[x] for x in self.test_set["label"][:self.num_examples]]

            test_inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                logits = model(**test_inputs).logits
            test_pred_labels = np.argmax(logits.numpy(), axis=1)

            accuracy = accuracy_score(test_true_labels, test_pred_labels)
            f1 = f1_score(test_true_labels, test_pred_labels, average='weighted')
            report = classification_report(test_true_labels, test_pred_labels, target_names=self.target_names, output_dict=True, digits=4)

            evaluation_results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "F1 Score": f1,
            })

            for i, name in enumerate(self.target_names):
                evaluation_results[-1][name + " F1"] = report[name]["f1-score"]

            ax = axes[idx // num_columns, idx % num_columns]
            conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
            im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f"{model_name}\nAccuracy: {accuracy:.2f}\nF1 Score: {f1:.2f}")
            tick_marks = np.arange(len(self.target_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(self.target_names, rotation=45, ha='right')
            ax.set_yticklabels(self.target_names)

            # Display F1 scores in the confusion matrix
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    text_color = "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
                    ax.text(j, i, f"{conf_matrix[i, j]}\n({report[self.target_names[i]]['f1-score']:.2f})",
                            ha="center", va="center", color=text_color, fontsize=8)

        for idx in range(num_models, num_rows * num_columns):
            axes[idx // num_columns, idx % num_columns].axis("off")

        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.set_index("Model", inplace=True)

        print(evaluation_df)
        plt.show()
        return evaluation_df


def reduce_dataset_size_and_split(dataset, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42):
    assert train_fraction + val_fraction + test_fraction <= 1, "The sum of the fractions should not exceed 1."

    np.random.seed(seed)
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_size = int(train_fraction * num_samples)
    val_size = int(val_fraction * num_samples)
    test_size = int(test_fraction * num_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]

    train_data = dataset.select(train_indices)
    val_data = dataset.select(val_indices)
    test_data = dataset.select(test_indices)

    return train_data, val_data, test_data


class TextClassificationDataset(Dataset):
    """
    This class is designed to transform text data into a format that is suitable for training transformer-based models,
    such as BERT. It inherits from PyTorch's Dataset class, making it compatible with PyTorch's DataLoader for efficient
    data loading.

    Attributes:
    data : DataFrame
    The dataset containing the text and their corresponding labels.
    tokenizer : transformers.PreTrainedTokenizer
    The tokenizer corresponding to the transformer model to be used.
    It will be used to convert text into tokens that the model can understand.
    max_length : int
    The maximum length of the sequences. Longer sequences will be truncated, and shorter ones will be padded.

    Methods:
    len()
    Returns the number of examples in the dataset.
    getitem(index)
    Transforms the text at the given index in the dataset into a format suitable for transformer models.
    It returns a dictionary containing the input_ids, attention_mask, and the label for the given text.

    """
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'][index]
        label = self.data['label'][index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

yf.pdr_override() # Modify yf with pdr

# Start for opoint sentiment data: 01/01/2018
# End for opoint sentiment data: 28/02/2023

def financial_dataset(stock, num_of_labels=3, cutoff=0.30,
                      start_date="2018-01-01", end_date="2023-03-01") :
    ''' Downloads financial data for a stock and process it in the desired format
        Parameters :
          stock(str) : The desired stock's code
          cutoff(float) : A float indicating a percentage under which no price change is considered
          increase or decrease eg. 0.25 = 0.25% price change from close-to-close
          num_of_labels(2 or 3) : Number of labels to use. 2 = [Increase,Decrease].
                                  3=[Increase, Decrease, Sideways]
          start_date(str) : "year-month-day" The day data collection will start .
          end_date(str) : "year-month-day" The day data collection will stop .    '''
    # parameter value check
    if (num_of_labels < 2 or num_of_labels > 3):
        return print('Number of labels can be either 2 or 3')

    fin_data = pdr.get_data_yahoo(stock, start=start_date, end=end_date)

    print(f"{stock} financial dataframe dimensions ", fin_data.shape)

    # initialize price_change column
    fin_data['Price_change'] = 1
    fin_data['date'] = 0
    dates = fin_data.index
    yesterday = str(dates[0].date())

    # How much should the price change in abs value to be considered increase/decrease.
    for date in dates[1:] :
        today = str(date.date())

        yesterday_pr = fin_data.loc[yesterday, 'Close']
        today_pr = fin_data.loc[today, 'Close']
        diff = 100 * (today_pr - yesterday_pr)/yesterday_pr

        if (num_of_labels == 3) :
            if (diff > cutoff) :
                # price increase
                price_change = +1
            elif (diff < -cutoff) :
                # price decrease
                price_change = -1
            else:
                # almost steady price
                price_change = 0
        elif (num_of_labels == 2 ):
            if (diff > 0 ) :
                # price increase
                price_change = +1
            elif (diff <= 0 ) :
                price_change = -1

        yesterday = today
        fin_data.loc[today,'Price_change'] = price_change
        fin_data.loc[today,'date'] = today

    incr = fin_data[fin_data['Price_change'] == 1 ].shape[0]
    decr = fin_data[fin_data['Price_change'] == -1 ].shape[0]
    stable = fin_data[fin_data['Price_change'] == 0 ].shape[0]
    print(f'Positive changes : {incr}')
    print(f'Negative changes : {decr}')
    print(f'No changes : {stable}')

    fin_data['ticker'] = stock

    fin_data.drop(columns=['date'], inplace=True)
    fin_data.reset_index(inplace=True)
    fin_data.rename(columns={'Date': 'date'}, inplace=True)

    fin_data = fin_data.drop(columns = ['Low', 'High', 'Adj Close'], axis=1)

    return fin_data


# Function to find the next business day
def next_business_day(date):
    next_day = date + pd.Timedelta(days=1)
    while next_day.weekday() in [5, 6] or next_day in market_holidays:
        next_day += pd.Timedelta(days=1)
    return next_day


def convert_week_and_holidays(news):
    # Convert the 'date' column to datetime dtype
    news['date'] = pd.to_datetime(news['date'])

    # Apply the function to shift news dates to the next business day for holidays, Saturdays, and Sundays
    news['date'] = news['date'].apply(lambda x: next_business_day(x) if x.weekday() in [5, 6] or x in market_holidays else x)

    return news

def read_news(ticker, data):
    ''' Reads news relevant to 'ticker' from the "raw_partner_headlines.csv" csv file.
        Returns a dataframe in the format :[ text | date | ticker  ] '''

    # Format the date column to match financial dataset
    news = data[data['ticker'] == ticker]

    print(f"Found {news.shape[0]} news regarding {stock} stock")
    return news

def merge_fin_news(df_fin, df_news, how='inner'):
    '''
    Merges the financial data dataframe with the news dataframe and rearranges the column order
    Parameters:
        df_fin: DataFrame containing financial data
        df_news: DataFrame containing news data
        how(str): Merging technique : 'inner', 'outer' etc.. (check pd.merge documentation)
    Returns:
        merged_df: Merged DataFrame
    '''
    # Convert the 'date' column in both DataFrames to datetime dtype
    df_fin['date'] = pd.to_datetime(df_fin['date'])
    df_news['date'] = pd.to_datetime(df_news['date'])

    # Merge on both 'date' and 'ticker' columns using the specified merging technique
    merged_df = df_fin.merge(df_news, on=['date', 'ticker'], how=how)

    return merged_df

def sentim_analyzer(df, tokenizer, model):
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to the detected device
    model.to(device)

    for i in tqdm(df.index):
        try:
            text = df.loc[i, 'text']
        except KeyError:
            return print('\'text\' column might be missing from dataframe')

        # Pre-process input text
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        # Move input tensors to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():  # Inference doesn't require gradient computation
            outputs = model(**inputs)

        # Compute softmax to get probabilities from logits
        predictions = softmax(outputs.logits, dim=-1).cpu()  # Move predictions back to CPU

        # Update dataframe with sentiment scores
        df.loc[i, 'negative'] = predictions[0][0].item()
        df.loc[i, 'neutral'] = predictions[0][1].item()
        df.loc[i, 'positive'] = predictions[0][2].item()

    # Rearrange column order
    try:
        df = df[['date', 'ticker', 'Open', 'Close', 'Volume', 'text', 'positive', 'negative', 'neutral', 'price_change']]
    except KeyError:
        pass

    return df


def merge_dates(df):
    '''
    Given a df that contains columns [date, stock, Open, Close, Volume, headline, Positive, Negative, Neutral, Price_change],
    take the average of Positive, Negative, Neutral sentiment scores for each date and return a df that contains each
    date exactly one time. The return df has no column 'headline' since the scores now refer to an average of multiple
    news headlines.
        Parameters :
          df : A dataframe with columns [date, stock, Open, Close, Volume, headline, Positive, Negative, Neutral, Price_change]
          returns df : aggragated sentiment scores by date with columns [date, stock, Open, Close, Volume, headline, Positive, Negative, Neutral, Price_change]
    '''

    # Take the average for Positive, Negative and Neutral columns by date. Drop headline column and all other columns per date are identical.
    dates_in_df = df['date'].unique()
    new_df = df.copy(deep=True).head(0)  # Just take the df structure with no data inside
    new_df = new_df.drop(columns=['text'])  # Drop headline column

    for date in dates_in_df:
        sub_df = df[df['date'] == date]  # Filter specific dates
        avg_positive = sub_df['positive'].mean()
        avg_negative = sub_df['negative'].mean()
        avg_neutral = sub_df['neutral'].mean()
        sub_df = sub_df.drop(columns=['text'])  # Drop text column

        stock = sub_df.iloc[0]['ticker']

        sub_df = sub_df.head(0)  # Empty sub_df to populate with just 1 row for each date
        sub_df.loc[0] = [date, stock, avg_positive, avg_negative, avg_neutral]  # Populate the row
        # Add sub_df's row to the new dataframe
        new_df = pd.concat([new_df, sub_df], axis=0, ignore_index=True)
    print(f" Dataframe now contains sentiment score for {new_df.shape[0]} different dates.")
    return(new_df)

#Get hits of sentiment and price changes
def mark_hits(group, PRICE_CHANGE_THRESHOLD=PRICE_CHANGE_THRESHOLD,SENTIMENT_THRESHOLD=SENTIMENT_THRESHOLD):
    threshold_percent = PRICE_CHANGE_THRESHOLD  # Define threshold for significant price change
    threshold_sentiment = SENTIMENT_THRESHOLD  # Define threshold for strong sentiment
    for i in range(len(group) - 3):
        for j in range(1, 4):  # Look 1, 2, 3 days ahead
            current_row = group.iloc[i]
            future_row = group.iloc[i + j]

            # Price field contains the actual price
            current_price = current_row['Close']
            future_price = future_row['Close']

            # Calculate the percentage price change
            price_percent_change = ((future_price - current_price) / current_price) * 100

            # Check for strong positive sentiment and significant price increase
            if current_row['positive'] > threshold_sentiment and price_percent_change > threshold_percent:
                group.at[group.index[i], 'hit'] = True
                break

            # Check for strong negative sentiment and significant price decrease
            elif current_row['negative'] > threshold_sentiment and price_percent_change < -threshold_percent:
                group.at[group.index[i], 'hit'] = True
                break

            else:
                group.at[group.index[i], 'hit'] = False

    return group

class FinancialDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['sentence']
        label = self.data.iloc[index]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
