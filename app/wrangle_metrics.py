import pandas as pd
import plotly.graph_objs as go



def return_metrics():
    """Creates plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """
    df_metrics = pd.read_excel('../data/metrics_file.xlsx')
    

    df_summary = pd.read_excel('../data/metrics_summary_file.xlsx')
    df_summary = df_summary.rename(columns={'Unnamed: 0':'Metric'})
    df_accuracy = df_summary[df_summary['Metric']=='mean']
    df_accuracy = df_accuracy.drop('Metric',axis=1)
    df_accuracy=df_accuracy.T

    
    graph_one = []
    graph_one.append(
      go.Bar(
      x = df_metrics.Column.tolist(),
      y = df_metrics.Accuracy.tolist()
      
      ,
      )
    )

    layout_one = dict(title = 'Accuracy by Category',
                xaxis = dict(title = 'Emergency Category',),
                yaxis = dict(title = 'Accuracy'),
               margin=dict(l=40, r=60, t=100, b=200)
                )
     
    graph_two = []
    graph_two.append(
      go.Bar(
      x = df_metrics.Column.tolist()  ,    
      y = df_metrics.Precision.tolist()   
            )
    )

    layout_two = dict(title = 'Precision by Category',
                xaxis = dict(title = 'Emergency Category',),
                yaxis = dict(title = 'Precision'),
                margin=dict(l=40, r=60, t=100, b=200)
                )
    
    graph_three = []
    graph_three.append(
      go.Bar(
      x = df_metrics.Column.tolist(),
      y = df_metrics.Accuracy.tolist(),
      )
    )

    layout_three = dict(title = 'Recall by Category',
                xaxis = dict(title = 'Emergency Category',),
                yaxis = dict(title = 'Recall'),
                margin=dict(l=40, r=60, t=100, b=200)
                )
     
    graph_four = []
    graph_four.append(
      go.Bar(
      x = df_metrics.Column.tolist(),
      y = df_metrics.Precision.tolist(),
      )
    )

    layout_four = dict(title = 'F1 Score by Category',
                xaxis = dict(title = 'Emergency Category',),
                yaxis = dict(title = 'F1 Score'),
                margin=dict(l=40, r=60, t=100, b=200)
                )
    



    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))


    return figures


