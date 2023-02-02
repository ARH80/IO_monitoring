import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd

time_scope_size = 10
address_scope_size = 25 * 1e6


class BLKPlotter:

    @staticmethod
    def get_features_dataframe():
        def parser(record):
            tokens = record.split()
            result = {
                'timestamp': float(tokens[3]),
                'time_scope': float(tokens[3]) // time_scope_size,
                'cid': int(tokens[1]),
                'sid': int(tokens[2]),
                'pid': int(tokens[4]),
                'action': tokens[5],
                'rw': tokens[6],
                'rw_spec': 'R' if 'R' in tokens[6] else 'W' if 'W' in tokens[6] else 'N',
                'start_address': int(tokens[7])
            }

            result.update({'n_sectors': int(tokens[9]) if '+' in tokens else 1})
            result.update({'size': result['n_sectors'] * 512})
            result.update({'end_address': result['start_address'] + result['size']})

            result.update({
                'scope_start_address': result['start_address'] // address_scope_size,
                'scope_end_address': result['start_address'] // address_scope_size
            })

            return result

        with open('parsed_trace.txt', 'r') as f:
            lines = f.readlines()
            dicts = [parser(x) for x in lines]

        df = pd.DataFrame(dicts)
        return df[df.rw_spec != 'N']

    @staticmethod
    def pie_plot(df, save=False):
        palette_color = sns.color_palette('bright')

        plt.pie(
            [df[df.rw_spec == 'R'].shape[0], df[df.rw_spec == 'W'].shape[0]],
            labels=['Read', 'Write'],
            colors=palette_color,
            autopct='%.0f%%')

        if save:
            plt.savefig('blk_rw_pie_plot.png')
        else:
            plt.show()

    @staticmethod
    def density_on_size(df, save=False):
        df.plot.hist(column=['size'], by='rw_spec', bins=50)

        if save:
            plt.savefig('blk_density_on_size_plot.png')
        else:
            plt.show()

    @staticmethod
    def rw_intensive_plot(df, save=True):
        def count_R(group):
            return group[group == 'R'].shape[0]

        def count_W(group):
            return group[group == 'W'].shape[0]

        dp = df.groupby(['time_scope']).agg(r_count=('rw_spec', count_R), w_count=('rw_spec', count_W))
        dp.plot.bar()

        if save:
            plt.savefig('blk_rw_intensive_plot.png')
        else:
            plt.show()

    @staticmethod
    def address_frequency(df, save=True):
        c = Counter()
        _ = df[['scope_start_address', 'scope_end_address']].apply(
            lambda row: c.update(list(range(row[0], row[1] + 1))), axis=1)

        scopes, counts = zip(*c.items())
        plt.bar(scopes, counts)

        if save:
            plt.savefig('blk_rw_intensive_plot.png')
        else:
            plt.show()



