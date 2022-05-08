import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class Resource:
    def __init__(self, id: str, weights: list, grades: list):
        self.id = id
        data = {
            "weight": weights,
            "grade": grades
        }
        sorted_info = self.sort_info(pd.DataFrame(data))
        
        self.info = self.calculate_metadata(sorted_info)
        self.calculate_heterogeneity()
       
    def sort_info(self, data: pd.DataFrame):
        data.sort_values(by='grade', ascending=False, inplace=True)
        data.reset_index(drop=True,inplace=True)
        
        return data
         
    def calculate_metadata(self, sorted_info):
        df = sorted_info
        df['cml_weight'] = df['weight'].cumsum()
    
        sum_weights = sum(df['weight'])
        df['weight_pct'] = np.divide(df['weight'], sum_weights)
        df['cml_weight_pct'] = df['weight_pct'].cumsum()
        df['yield'] = np.multiply(df['weight'], df['grade'])
        df['cml_yield'] = df['yield'].cumsum()
        df['cml_grade'] = np.divide(df['cml_yield'], df['cml_weight'])
        
        sum_yield = sum(df['yield'])
        df['recovery'] = np.divide(df['yield'], sum_yield)
        df['cml_recovery'] = df['recovery'].cumsum()
        
        return df
    
    def calculate_heterogeneity(self):
        df = self.info
        
        b = df['weight']
        c = df['cml_grade'].iloc[-1]
        d = sum(df['weight'])
        a = np.subtract(df['grade'], c)
        
        num, den = np.multiply(a, b), np.multiply(c, d)
        df['dist_het'] = np.power(np.divide(num, den), 2)  # (num/den)^2
        self.cons_het = len(df) * sum(df['dist_het'])
        
    def plot_grade_recovery_curve(self, ax: plt.Figure):
        s = 3
        
        df = self.info
        ax_sec = ax.twinx()

        x = df['cml_weight_pct']
        ax.set_xlabel("Cumulative Mass %")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        y = df['cml_grade']
        ax.set_ylabel(f"Cumulative {self.id} Grade", c="blue") 

        ax.plot(x, y, color='blue', alpha=0.2, ls='--')
        ax.scatter(x, y, color='blue', s=s)


        y = df['cml_recovery']
        ax_sec.set_ylabel(f"Cumulative {self.id} Recovery", c="red") 
        ax_sec.set_ylim(0, 1)
        ax_sec.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_sec.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        ax_sec.plot([0, 1], [0, 1], color='grey', alpha=0.2, ls='-.')

        ax_sec.plot(x, y, color='red', alpha=0.2, ls='--')
        ax_sec.scatter(x, y, color='red', s=s)

        y_prime = np.subtract(y, x)
        ax_sec.plot(x, y_prime, color='red', alpha=0.2, ls=':')

        x_max, y_max = x[np.argmax(y_prime)], y[np.argmax(y_prime)]
        ax_sec.plot([x_max, x_max], [0, y_max], c="black", ls=':', alpha=0.35)
        ax_sec.annotate(f"{x_max * 100:.3}%", (x_max, 0), xytext=(5, 10), textcoords='offset points', rotation=90)
        ax_sec.plot([x_max, 1], [y_max, y_max], c="black", ls=':', alpha=0.35)
        ax_sec.annotate(f"{y_max * 100:.3}%", (1, y_max), xytext=(-35, -15), textcoords='offset points')



class Grouping:
    def __init__(self, id):
        self.id = id
        self.resources = {}
        
    def calculate_resource_heterogeneity(self, resource_id: str, weights: list, grades: list):
        if len(weights) - len(grades) == 0:
            self.resources[resource_id] = Resource(resource_id, weights, grades)
        else:
            raise Exception("Weight and grade arrays must be same length")
        
    def plot_grade_recovery_curves(self):
        fig, axs = plt.subplots(len(self.resources), figsize=(8, (6 * len(self.resources))))
        
        # TODO: Fix this to better handle one or more resources
        if len(self.resources) == 1:
            axs.set_title(f'ID: {self.id}')
            
            for i, resource_id in enumerate(self.resources.keys()):
                self.resources[resource_id].plot_grade_recovery_curve(axs)
        else:
            axs[0].set_title(f'ID: {self.id}')

            for i, resource_id in enumerate(self.resources.keys()):
                self.resources[resource_id].plot_grade_recovery_curve(axs[i])

        return fig
    