import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import Counter
from datetime import datetime

fig, ax = plt.subplots(2, 2, frameon=False, figsize=(10, 10))


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
                'scope_start_address': int(result['start_address'] // address_scope_size),
                'scope_end_address': int(result['start_address'] // address_scope_size)
            })

            return result

        with open('parsed_trace.txt', 'r') as f:
            lines = f.readlines()
            dicts = [parser(x) for x in lines]

        df = pd.DataFrame(dicts)
        return df[df.rw_spec != 'N']

    @staticmethod
    def pie_plot(df):
        palette_color = sns.color_palette('bright')

        ax[0, 0].pie(
            [df[df.rw_spec == 'R'].shape[0], df[df.rw_spec == 'W'].shape[0]],
            labels=['Read', 'Write'],
            colors=palette_color,
            autopct='%.0f%%')

    @staticmethod
    def density_on_size(df):
        ax[0, 1].plot.hist(column=['size'], by='rw_spec', bins=50)

    @staticmethod
    def rw_intensive_plot(df):
        def count_R(group):
            return group[group == 'R'].shape[0]

        def count_W(group):
            return group[group == 'W'].shape[0]

        df = df.groupby(['time_scope']).agg(r_count=('rw_spec', count_R), w_count=('rw_spec', count_W))
        df.plot.bar(ax=ax[1, 0])

    @staticmethod
    def address_frequency(df):
        c = Counter()
        _ = df[['scope_start_address', 'scope_end_address']].apply(
            lambda row: c.update(list(range(row[0], row[1] + 1))), axis=1)

        scopes, counts = zip(*c.items())
        plt.bar(scopes, counts, ax=ax[1, 1])


dp = BLKPlotter.get_features_dataframe()

time_scope_size = int(dp.timestamp.max() // 100)
address_scope_size = 25 * 1e6

BLKPlotter.address_frequency(dp)
BLKPlotter.rw_intensive_plot(dp)
BLKPlotter.pie_plot(dp)
BLKPlotter.density_on_size(dp)
fig.savefig('blk-4main-plots.png')
