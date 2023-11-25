from matplotlib import pyplot as plt


class LineChart:


    def __init__(self, data, title, x_label, y_label, output_file_name):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.out_file = output_file_name
        self.df = data

    def plot(self):
        fig, ax = plt.subplots()

        ax.set_title(self.title)
        ax.set_ylabel(self.y_label)
        ax.set_xlabel(self.x_label)

        for column in self.df:
            ax.plot(self.df[column], data=self.df)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return fig, ax

    def save_plot(self):
        print("..: Saving plot in: ", self.out_file)
        plt.savefig(self.out_file)
        plt.close()
