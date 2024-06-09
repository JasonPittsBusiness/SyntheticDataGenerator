import pandas as pd
from sdv.single_table import CTGANSynthesizer #Synthesizer
from sdv.metadata import SingleTableMetadata #Metadata gen
import matplotlib.pyplot as plt #Plotting
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport #Data reports
import seaborn as sns #Matrix generation

#Load file to DF and remove ID column
def load_data(filepath):
    data = pd.read_csv(filepath)
    if 'User ID' in data.columns:
        data.drop('User ID', axis=1, inplace=True)
    return data

def setup_metadata(data):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.visualize(show_table_details='summarized', output_filepath='/PathTo/my_metadata.png')
    return metadata

#Adjust epochs and other data to tune outcome
#Learn more here https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer
def synthesize_data(metadata, data, filepath):
    synthesizer = CTGANSynthesizer(metadata, enforce_rounding=False, epochs=200, verbose=True, cuda=True)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows=len(data), batch_size=400, max_tries_per_batch=100)
    synthetic_data.to_csv(filepath, index=False)
    return synthesizer, synthetic_data

#Plot the loss data
def plot_loss_data(synthesizer):
    lossdata = synthesizer.get_loss_values()
    lossdata['Generator Loss'] = lossdata['Generator Loss'].astype(float)
    lossdata['Discriminator Loss'] = lossdata['Discriminator Loss'].astype(float)

    plt.figure(figsize=(30, 8))
    plt.plot(lossdata['Epoch'], lossdata['Generator Loss'], label='Generator Loss')
    plt.plot(lossdata['Epoch'], lossdata['Discriminator Loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss Over Epochs')
    plt.legend()
    plt.show()

#Ensure data validity and structure, these should be 100%
def generate_reports(data, synthetic_data, metadata):
    metadata_dict = metadata.to_dict()
    diagnostic_report = DiagnosticReport()
    diagnostic_report.generate(data, synthetic_data, metadata_dict)
    
    quality_report = QualityReport()
    quality_report.generate(data, synthetic_data, metadata_dict, verbose=False)
    print(quality_report.get_score())
    print(quality_report.get_properties())

#Correlation Matrix Gen, adjust tuning for accurate correlation
def plot_correlation(data, synthetic_data):
    data_corr = data[['Age', 'EstimatedSalary', 'Purchased']].corr()
    syn_corr = synthetic_data[['Age', 'EstimatedSalary', 'Purchased']].corr()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    sns.heatmap(data_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation'}, ax=axes[0])
    axes[0].set_title('Correlation Matrix - Real Data')
    sns.heatmap(syn_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation'}, ax=axes[1])
    axes[1].set_title('Correlation Matrix - Synthetic Data')
    plt.tight_layout()
    plt.show()

def main():
    data_path = '/PathTo/Customer_Behaviour.csv'
    synthetic_data_path = '/PathTo/synthetic_customer_data.csv'
    
    data = load_data(data_path)
    if input("Continue to metadata setup? (yes/no) ").lower() != 'yes':
        return
    
    metadata = setup_metadata(data)
    if input("Continue to data synthesis? (yes/no) ").lower() != 'yes':
        return
    
    synthesizer, synthetic_data = synthesize_data(metadata, data, synthetic_data_path)
    if input("Continue to plot lossdata? (yes/no) ").lower() != 'yes':
        return
    
    plot_loss_data(synthesizer)
    if input("Continue to generate reports? (yes/no) ").lower() != 'yes':
        return
    
    generate_reports(data, synthetic_data, metadata)
    if input("Continue to plot correlation? (yes/no) ").lower() != 'yes':
        return
    
    plot_correlation(data, synthetic_data)

if __name__ == "__main__":
    main()