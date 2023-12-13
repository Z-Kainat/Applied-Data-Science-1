import pandas as pd
import matplotlib.pyplot as plt

def plot_mean_bp_and_hr_by_age(df):
    """
    Plot the mean Blood Pressure (BP) and Maximum Heart Rate (HR) by age.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    None
    """
    age_mean_bp = df.groupby('Age')['BP'].mean()
    age_mean_HR = df.groupby('Age')['Max HR'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(age_mean_bp.index, age_mean_bp.values, label='Mean BP')
    plt.plot(age_mean_HR.index, age_mean_HR.values, label='Mean HR')
    plt.xlabel('Age')
    plt.ylabel('Count / Mean BP')
    plt.title('Mean BP and HR by Age')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_distribution_of_heart_disease(df):
    """
    Plot the distribution of Heart Disease in the dataset.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    None
    """
    heart_disease_count = df['Heart Disease'].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(heart_disease_count, labels=heart_disease_count.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Heart Disease')
    plt.show()

def plot_distribution_of_heart_disease_by_sex(df):
    """
    Plot the distribution of Heart Disease by gender (Sex) in the dataset.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    None
    """
    heart_disease_by_sex = pd.crosstab(df['Sex'], df['Heart Disease'])

    plt.figure(figsize=(8, 6))
    heart_disease_by_sex.plot(kind='bar', stacked=False)
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.title('Distribution of Heart Disease by Sex')
    plt.legend(title='Heart Disease')
    plt.show()

# Example usage
df = pd.read_csv(r'D:\fiverr\assingment2\Heart_Disease_Prediction.csv')
plot_mean_bp_and_hr_by_age(df)
plot_distribution_of_heart_disease(df)
plot_distribution_of_heart_disease_by_sex(df)
