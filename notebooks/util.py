# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import itertools
from operator import itemgetter
import xgboost as xgb
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import warnings

WIDTH = 0.4

COLORS = {
    'scudra_blue': '#0000FF',
    'scudra_gray': '#BEBEBE',
    'scudra_lightblue': '#8CAFE6',
    'scudra_darkgray': '#646464',
    'white': '#FFFFFF',
    'black': '#000000',
    'orange': '#d9a957',
    'light_orange': '#FFC293',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'skyblue': '#4FAFFF',
    'cobalt': '#4D71AF',
    'dark_skyblue': '#096EC1',
    'red': '#D55E00',
    'purple': '#C41B79'
}

PALETTES = {
    'default': {
        'primary_a': COLORS['cobalt'],
        'primary_b': COLORS['skyblue'],
        'secondary_a': COLORS['orange'],
        'secondary_b': COLORS['light_orange'],
        'aux_a': COLORS['purple'],
        'aux_b': None,
    },
    'scudra': {
        'primary_a': COLORS['scudra_blue'],
        'primary_b': COLORS['scudra_lightblue'],
        'secondary_a': COLORS['scudra_darkgray'],
        'secondary_b': COLORS['scudra_gray'],
        'aux_a': COLORS['green'],
        'aux_b': None,
    }
}


def xgbCV(train_X, train_y, test_X, eta=[0.05],max_depth=[6],sub_sample=[0.9],colsample_bytree=[0.9]):
    #train_y = train_df['SeriousDlqin2yrs'] # label for training data
    #train_X = train_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False) # feature for training data
    #test_X = test_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False) # feature for testing data
    
    skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True) # stratified sampling
    train_performance ={} 
    val_performance={}
    for each_param in itertools.product(eta,max_depth,sub_sample,colsample_bytree): # iterative over each combination in parameter space
        xgb_params = {
                    'eta':each_param[0],
                    'max_depth':each_param[1],
                    'sub_sample':each_param[2],
                    'colsample_bytree':each_param[3],
                    'objective':'binary:logistic',
                    'eval_metric':'auc',
                    'silent':0
                    }
        best_iteration =[]
        best_score=[]
        training_score=[]
        for train_ind,val_ind in skf.split(train_X,train_y): # five fold stratified cross validation
            X_train,X_val = train_X.iloc[train_ind,],train_X.iloc[val_ind,] # train X and train y
            y_train,y_val = train_y.iloc[train_ind],train_y.iloc[val_ind] # validation X and validation y
            dtrain = xgb.DMatrix(X_train,y_train,feature_names = X_train.columns) # convert into DMatrix (xgb library data structure)
            dval = xgb.DMatrix(X_val,y_val,feature_names = X_val.columns) # convert into DMatrix (xgb library data structure)
            model = xgb.train(xgb_params,dtrain,num_boost_round=1000, 
                              evals=[(dtrain,'train'),(dval,'val')],verbose_eval=False,early_stopping_rounds=30) # train the model
            best_iteration.append(model.attributes()['best_iteration']) # best iteration regarding AUC in valid set
            best_score.append(model.attributes()['best_score']) # best score regarding AUC in valid set
            #training_score.append(model.attributes()['best_msg'].split()[1][10:]) # best score regarding AUC in training set
        valid_mean = (np.asarray(best_score).astype(np.float).mean()) # mean AUC in valid set
        train_mean = (np.asarray(training_score).astype(np.float).mean()) # mean AUC in training set
        val_performance[each_param] =  train_mean
        train_performance[each_param] =  valid_mean
        print ("Parameters are {}. Training performance is {:.4f}. Validation performance is {:.4f}".format(each_param,train_mean,valid_mean))
    print('\nBest parameters: ', sorted(train_performance.items(), reverse=True, key=itemgetter(1))[0], '\n')
    return train_performance



def get_resampling(X, y, verbose=False, random_state=17, by='oversampling_random', good_mult=1.0):
    X_resampled = pd.DataFrame()
    y_resampled=pd.Series()
    
    columns = X.columns
    
    if verbose:
        print('[get_resampling] Initial class distribution:\n%s'%(y.value_counts()))
        print('[get_resampling] Executing resampling approache:', by)
    
    if by=='undersampling':
        '''
        training_set = X.copy()
        training_set['bom_pagador']=y
        sample_size_bad = training_set[training_set['bom_pagador'] == 0]
        sample_size_good = int(good_mult * sample_size_bad)
        Xgood = training_set[training_set['bom_pagador'] == 1].sample(n = sample_size_good, random_state=random_state)
        Xbad  = training_set[training_set['bom_pagador'] == 0].sample(n = sample_size_bad, random_state=random_state)
        training_set = pd.concat([Xbad, Xgood])
        X_resampled = training_set.iloc[:, :-1].values
        y_resampled = training_set['bom_pagador'].values
        '''
        X_resampled, y_resampled = RandomUnderSampler(random_state=random_state).fit_resample(X, y)
    elif by == 'oversampling_random':
        X_resampled, y_resampled = RandomOverSampler(random_state=random_state).fit_resample(X, y)
    elif by == 'oversampling_smote':
        X_resampled, y_resampled = SMOTE(random_state=random_state, n_jobs=8).fit_resample(X, y)
    elif by == 'oversampling_adasyn':
        X_resampled, y_resampled = ADASYN(random_state=random_state).fit_resample(X, y)
    else:
        raise Exception('by == {} is not valid'.format(by))
        
    X_resampled = pd.DataFrame(X_resampled, columns=columns)
    y_resampled = pd.Series(y_resampled, name='bom_pagador')
    
    if verbose:
        print('\nFinal class distribution:\n%s'%(y_resampled.value_counts()))
        
    #X_resampled['bom_pagador'] = y_resampled
    return X_resampled, y_resampled

def prob_scale(x):
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin)

class PSI:
    def __init__(self,
                 dist_a: np.array,
                 dist_b: np.array,
                 target_a: np.array = None,
                 target_b: np.array = None,
                 bucket_type: str = 'bins',
                 n_buckets: int = 10,
                 fixed_limits: bool = False,
                 color_palette: str = 'default'):
        """This class calculates and plots PSI given two reference
        distributions.

        Parameters
        ----------
        dist_a : np.array
            First distribution to be compared.
        dist_b : np.array
            Second distribution to be compared.
        target_a : np.array, optional
            Targets of the first distribution, by default None.
            If none, than it is assumed that no default rate should be plotted
            for this distribution.
        target_b : np.array, optional
            Targets of the second distribution, by default None
            If none, than it is assumed that no default rate should be plotted
            for this distribution.
        bucket_type : str, optional
            Describes graph's type and must be 'bins' or 'quantiles'.
            By default 'bins'.
        fixed_limits : bool, optional
            Only useful when the `bucket_type` parameter is 'bins':
                - If 'True', it'll lock necessarily the distribution's range
                between 0 and 100.
                - If 'False', then the limits going to be the min and max
                distribution values, respectively.

            Note: Commonly, in PSI calculus the limits naturally are the
            min/max values of the distribution. But it'spossible to lock
            the range between 0 and 100 to only then get the buckets and
            perform the PSI.
        n_buckets : int, optional
            Quantity of bins groups, by default 10.
        color_palette : str, optional
            Graph style, by default 'default'. You can choose between:
                - `default`: Default backtests color scheme
                - `scudra`: Scudra official color scheme
        """
        self._dist_validations([dist_a, dist_b], [target_a, target_b])

        self.__bucket_type = bucket_type
        self.__n_buckets = n_buckets
        self.__breakpoints = self.calculate_breakpoints(
            self.__bucket_type, self.__n_buckets, fixed_limits, dist_a)
        self.__dist_qtd, self.__dist_perc = self.calculate_distribution(
            dist_a, dist_b, self.__breakpoints)
        self.__default_rate_a = self.calculate_default_rate(
            dist_a, target_a, self.__breakpoints)
        self.__default_rate_b = self.calculate_default_rate(
            dist_b, target_b, self.__breakpoints)
        self.__value_list = self.calculate_psi(self.__dist_qtd,
                                               self.__dist_perc)
        self.__value = self.__value_list.sum()
        self.__palette = color_palette

    def _dist_validations(self, dists: list, targets: list):
        for name, dist, target in zip(['A', 'B'], dists, targets):
            if np.count_nonzero(np.isnan(dist)):
                raise AttributeError('Dist %s with NaN values!' % name)
            if len(dist.shape) != 1:
                raise AttributeError('Dist %s must be an array 1-d' % name)
            if target is not None and np.count_nonzero(np.isnan(target)):
                raise AttributeError('Target %s with NaN values!' % name)
            if target is not None and len(target.shape) != 1:
                raise AttributeError('Target %s must be an array 1-d' % name)

    def scale_range(self, input_values: np.array, min_value: float,
                    max_value: float) -> np.array:
        """Normalizes the `input_values` putting that given distribuition
        between `min_value` and `max_value`.

        Parameters
        ----------
        input_values : np.array
            Raw distribuition to be normalized.
        min_value : float
            Lower bound of the new distribuition.
        max_value : float
            Upper bound of the new distribuition.

        Returns
        -------
        np.array
            Returns the given array adjusted between the the lower and upper
            bound.
        """
        input_values = input_values.astype(float)
        input_values += -(np.min(input_values))
        input_values /= np.max(input_values) / (max_value - min_value)
        input_values += min_value
        return input_values

    def calculate_breakpoints(self, bucket_type: str, n_buckets: int,
                              fixed_limits: bool, dist: np.array) -> np.array:
        """Calculates the boundaries of a distribution based on bucket type.
            Arranges the number of elements per bucket.

            Parameters
            ----------
            bucket_type : str
                bucket type (bins or quantiles).
                If 'bins', separates distribution based on fixed ranges.
                If 'quantiles', separates distribution based on percentiles.
            n_buckets : int
                Quantity of bins groups
            fixed_limits : bool
                If 'True', it'll lock the distribution's range in 0 and 100
                If 'False', then the distribution limits gonna be got
                dynamically.

                Note: Only useful when the `bucket_type` parameter is 'bins'.
            dist : np.array
                Distribution of interest.

            Returns
            -------
            np.array
                Breakpoints of distribution.
        """
        breakpoints = np.arange(0, n_buckets + 1) / (n_buckets) * 100

        if bucket_type == 'bins':
            breakpoints = np.array(
                [i for i in np.arange(0., 1., 1. / n_buckets)] +
                [1]) if fixed_limits else self.scale_range(
                    breakpoints, np.min(dist), np.max(dist))
        elif bucket_type == 'quantiles':
            breakpoints = np.stack(
                [np.percentile(dist, b) for b in breakpoints])
        return breakpoints

    def calculate_distribution(self, dist_a: np.array, dist_b: np.array,
                               breakpoints: np.array) -> tuple:
        """Adjusts both distribuitions (`dist_a` and `dist_b`) according to the
        given breakpoints.

        Parameters
        ----------
        dist_a : np.array
            First distribution to be adjusted.
        dist_b : np.array
            Second distribution to be adjusted.
        breakpoints : np.array
            Breakpoints calculated by `calculate_breakpoints` method.

        Returns
        -------
        Tuple of lists. The fisrst list refers to `dist_a` as well the
        second one refers to the `dist_b`.
            (list of amounts of scores per bucket,
             list of percentages of scores per bucket)
        """

        dist_qtd = []  # quantidade de scores por bucket
        dist_perc = []  # porcentagem de scores por bucket

        for dist in [dist_a, dist_b]:
            qtd = np.histogram(dist, breakpoints)[0]
            perc = qtd / dist.shape[0]
            perc[perc == 0] = 0.0001

            dist_qtd.append(qtd)
            dist_perc.append(perc)
        return dist_qtd, dist_perc

    def calculate_default_rate(self, dist: np.array, targets: np.array,
                               breakpoints: np.array) -> np.array:
        """Calculates default rate for each bins groups (buckets).

        Parameters
        ----------
        dist : np.array
            Distribution of scores whose default rate must be calculated
        targets : np.array
            Targets of distribution. 0 means bad payer, 1 means good payer.
            A target value in index i refers to the score in the same index
            position in the score distribution.
        breakpoints : np.array
            Breakpoints calculated by `calculate_breakpoints` method.

        Returns
        -------
        np.array
            Default rate for each bucket in score distribution.
            Returns None if target is None.
        """

        if targets is not None:
            defaulter_rate = np.zeros(breakpoints.shape[0] - 1)

            for i in range(1, breakpoints.shape[0] - 1):
                bucket_index = ((dist >= breakpoints[i - 1])
                                & (dist < breakpoints[i]))
                defaulter_rate[i -
                               1] = 1 - targets[bucket_index].mean() if len(
                                   targets[bucket_index]) > 0 else np.nan

            bucket_index = dist >= breakpoints[-2]
            defaulter_rate[-1] = 1 - targets[bucket_index].mean() if len(
                targets[bucket_index]) > 0 else 0
            defaulter_rate *= 100

            return np.nan_to_num(defaulter_rate)
        else:
            return None

    def calculate_psi(self, dist_qtd: list, dist_perc: list) -> np.array:
        """Calculates PSI given an amount of scores per bucket and
        a percentage of scores per bucket.

        Parameters
        ----------
        dist_qtd : list
            List of amounts of scores per bucket.
        dist_perc : list
            List of percentage of scores per bucket.

        Returns
        -------
        np.array
            An array containing the partial PSI for each bucket.
            To get the final PSI value you should sum these values.
        """
        return (dist_perc[1] - dist_perc[0]) * np.log(
            dist_perc[1] / dist_perc[0])

    def plot(self,
             label_a: str = 'A',
             label_b: str = 'B',
             title: str = 'Population Stability Index (PSI): {psi:.4f}',
             ax: plt.axes = None,
             fontsize: int = 12,
             tight: bool = False,
             figsize: tuple = (20,5)) -> plt.axes:
        """Plots both distributions (`dist_a` and `dist_b`) adjusted
        according to the breakpoints. If targets are available, it'll  show the
        default rate per bucket as well.

        Parameters
        ----------
        label_a : str, optional
            Label for distribution a, by default 'A'
        label_b : str, optional
            Label for distribution b, by default 'B'
        title : str, optional
            Figure title,
            by default 'Population Stability Index (PSI): {psi:.4f}'
        ax : plt.axes, optional
            Axis to plot the PSI graph,
            by default None

        Returns
        -------
        plt.axes
            PSI Graph
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, facecolor=(1, 1, 1))

        ax.set_title(title.format(psi=self.value), {'size': fontsize})
        ax.set_xlabel('Scores', {'size': fontsize})
        ax.set_ylabel('Population (%)', {'size': fontsize})

        x_axis = np.arange(0.1, self.n_buckets + .1)
        ax.set_xticks(x_axis)

        if tight or figsize != (20,5):
            ax.set_xticklabels([
                "[{:.0f})".format(100 * self.breakpoints[i])
                for i in range(1, self.breakpoints.shape[0])
            ], {'size': fontsize - 2})
        else:
            ax.set_xticklabels([
                "[{:.0f}, {:.0f})".format(100 * self.breakpoints[i - 1],
                                          100 * self.breakpoints[i])
                for i in range(1, self.breakpoints.shape[0])
            ], {'size': fontsize - 2})

        ax.bar(x_axis - (WIDTH / 2),
               self.dist_perc[0] * 100,
               WIDTH,
               color=PALETTES[self.__palette]['primary_b'],
               label=label_a)
        ax.bar(x_axis + (WIDTH / 2),
               self.dist_perc[1] * 100,
               WIDTH,
               color=PALETTES[self.__palette]['secondary_b'],
               label=label_b)

        ax.legend(loc='best')

        if (self.default_rate_a is not None) or (self.default_rate_b
                                                 is not None):
            ax2 = ax.twinx()

            if tight:
                ax2.get_yaxis().set_visible(False)
            else:
                ax2.set_ylabel('Delinquency rate (%)', {'size': fontsize - 2})

            if self.default_rate_a is not None:
                ax2.plot(x_axis,
                         self.default_rate_a,
                         'o-',
                         linewidth=3,
                         color=PALETTES[self.__palette]['primary_a'])

            if self.default_rate_b is not None:
                ax2.plot(x_axis,
                         self.default_rate_b,
                         'o-',
                         linewidth=3,
                         color=PALETTES[self.__palette]['secondary_a'])
        return ax

    # Getters
    @property
    def n_buckets(self):
        return self.__n_buckets

    @property
    def bucket_type(self):
        return self.__bucket_type

    @property
    def breakpoints(self):
        return self.__breakpoints

    @property
    def dist_qtd(self):
        return self.__dist_qtd

    @property
    def dist_perc(self):
        return self.__dist_perc

    @property
    def default_rate_a(self):
        return self.__default_rate_a

    @property
    def default_rate_b(self):
        return self.__default_rate_b

    @property
    def value(self):
        return self.__value

    @property
    def values(self):
        return self.__value_list

class ROC_AUC:
    def __init__(self,
                 y_probas: np.array,
                 y_true: np.array,
                 average: str = 'macro',
                 color_palette: str = 'default'):
        """Generates the ROC curves from labels and predicted scores/probabilities

        Features:
            - Compute Area Under the Receiver Operating Characteristic Curve
        (ROC AUC) from prediction scores.
            - Find the optimal probability cutoff point for a classification model
        related to event rate

        Note: This implementation is restricted to the binary classification
        task or multilabel classification task in label indicator format.

        Parameters
        ----------
        y_probas : array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the
            positive class, confidence values, or non-thresholded measure of
            decisions (as returned by “decision_function” on some classifiers).
        y_true : array-like of shape (n_samples,)
            True binary labels or binary label indicators.
        average : str, optional
            If `None`, the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the data,
            must be:
                - 'micro': Calculate metrics globally by considering each
                element of the label indicator matrix as a label.
                - 'macro': Calculate metrics for each label, and find their
                unweighted mean. This does not take label imbalance into
                account.
        color_palette : str, optional
            Graph style, by default 'default'. You can choose between:
                - `default`: Default backtests color scheme
                - `scudra`: Scudra official color scheme
        """
        self.y_probas = y_probas
        self.y_true = y_true
        self.palette = color_palette
        self.fpr, self.tpr, self._auc = dict(), dict(), dict()
        self.average = average
        self._perform_roc()

    @property
    def y_probas(self):
        return self.__y_probas

    @y_probas.setter
    def y_probas(self, value: np.array):
        value = np.array(value)
        if np.count_nonzero(np.isnan(value)) or value is None:
            raise AttributeError('Found NaN values in `y_probas`!')
        elif ~(
            (value != 0) &
            (value != 1)).any() or value.dtype == int or len(value.shape) != 1:
            raise ValueError(
                'You must pass an float 1-d array of probabilites to `y_probas`!'
            )
        else:
            self.__y_probas = np.column_stack((1 - value, value))

    @property
    def y_true(self):
        return self.__y_true

    @y_true.setter
    def y_true(self, value: np.array):
        value = np.array(value)
        if np.count_nonzero(np.isnan(value)) or value is None:
            raise AttributeError('Found NaN values in `y_probas`!')
        elif ((value != 0) & (value != 1)).any():
            raise ValueError('You must pass an binary array to `y_true`!')
        else:
            self.__y_true = value

    @property
    def threshold(self):
        fpr, tpr, thr = roc_curve(self.y_true, self.y_probas[:, 1])
        true_negative_rate = 1 - fpr
        true_values = tpr + true_negative_rate
        return thr[np.argsort(np.absolute(true_values))][-1:][0]

    @property
    def palette(self):
        return self.__palette

    @palette.setter
    def palette(self, value: str):
        if value in PALETTES.keys():
            self.__palette = value
        else:
            raise ValueError('Palette not found! Choose one between [`' +
                             '`, `'.join(list(PALETTES.keys())) + '`]!')

    @property
    def value(self):
        return self._auc[self.average]

    @property
    def gini(self):
        return 2 * self.value - 1

    def _perform_roc(self):
        n_classes = self.y_probas.shape[1]

        for i in range(n_classes):
            self.fpr[i], self.tpr[i], _ = roc_curve(self.y_true,
                                                    self.y_probas[:, i],
                                                    pos_label=np.unique(
                                                        self.y_true)[i])
            self._auc[i] = auc(self.fpr[i], self.tpr[i])

        # micro
        self.fpr['micro'], self.tpr['micro'], _ = roc_curve(
            np.tile(self.y_true, 2), self.y_probas.ravel())
        self._auc['micro'] = auc(self.fpr['micro'], self.tpr['micro'])

        # macro
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        self.fpr['macro'] = np.unique(
            np.concatenate([self.fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        self.tpr['macro'] = np.zeros_like(self.fpr['macro'])
        for i in range(n_classes):
            self.tpr['macro'] += np.interp(self.fpr['macro'], self.fpr[i],
                                           self.tpr[i])

        # Finally average it and compute AUC
        self.tpr['macro'] /= n_classes
        self._auc['macro'] = auc(self.fpr['macro'], self.tpr['macro'])

    def plot(self,
             title: str = None,
             fontsize: int = 12,
             show_classes: bool = False,
             classes_legend=['Class 0', 'Class 1'],
             legend_loc: str = 'lower right',
             ax: matplotlib.axes.Axes = None,
             tight: bool = False,
             figsize: tuple = (10,7)):
        """Make all the magic happens to plot the ROC curve =)

        Parameters
        ----------
        title : str, optional
            Figure title, by default None
            If None, then the class parameters gonna be used to generate
            an appropriated title.
        fontsize : int, optional
            Text font size, by default 14
        show_classes : bool, optional
            If True it'll plot the ROC of each class, by default False
        classes_legend : list, optional
            Classes labels, by default ['Class 0', 'Class 1']
        legend_loc : str, optional
            The location of the legend, by default 'lower right'
                - The strings 'upper left', 'upper right', 'lower left',
                'lower right' place the legend at the corresponding
                corner of the axes/figure.
                - The strings 'upper center', 'lower center', 'center left',
                'center right' place the legend at the center of the
                corresponding edge of the axes/figure.
                - The string 'center' places the legend at the center of the
                axes/figure.
                - The string 'best' places the legend at the location, among
                the nine locations defined so far, with the minimum overlap
                with other drawn artists. This option can be quite slow for
                plots with large amounts of data; your plotting speed may
                benefit from providing a specific location.
        ax : matplotlib.axes.Axes, optional
            The axes on which the plot was drawn, by default None
        """

        if ax is None:
            fig, ax = plt.subplots(figsize= figsize,
                                   facecolor=(1, 1, 1),
                                   dpi=60)

        ax.plot([0, 1], [0, 1], 'k--', lw=2, color='gray', label='Random')
        ax.plot(self.fpr[self.average],
                self.tpr[self.average],
                lw=4,
                label='{} ROC - AUC {:.2%}'.format(self.average.title(),
                                                   self.value),
                color=PALETTES[self.palette]['aux_a'])
        ax.fill_between(self.fpr[self.average],
                        self.tpr[self.average],
                        facecolor=PALETTES[self.palette]['aux_a'],
                        alpha=0.15)

        _colors = ['secondary_a', 'primary_a']
        if show_classes:
            for i in range(self.y_probas.shape[1]):
                ax.plot(self.fpr[i],
                        self.tpr[i],
                        lw=3,
                        label='{} - AUC {:.2%}'.format(classes_legend[i],
                                                       self._auc[i]),
                        color=PALETTES[self.palette][_colors[i]])

        if title is None:
            title = 'ROC Curve - AUC {:.2%}; GINI {:.2%}'.format(
                self.value, self.gini)

        ax.set_title(title, fontsize=fontsize)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.minorticks_on()
        ax.grid(which='both', axis='both', color='lightgray')

        if not tight:
            ax.set_xlabel('False Positive Rate', fontsize=fontsize - 2)
            ax.set_ylabel('True Positive Rate', fontsize=fontsize - 2)

        ax.tick_params(labelsize=fontsize - 2)
        ax.legend(loc=legend_loc, fontsize=fontsize - 2)


class ScoreDistribution:
    def __init__(self,
                 dist: np.array,
                 target: np.array = None,
                 accumulated_defaulter_rate: bool = False,
                 buckets_as_quantiles: bool = False,
                 n_buckets: int = 10,
                 color_palette: str = 'default'):
        """This class plots a given distribution as histogram
        (in quantiles style or not) allowing show the defaulter rate for each
        bin.

        Parameters
        ----------
        dist : np.array
            Score distribution to be plotted as histogram
        target : np.array, optional
            Target distribution to be used to calculate the defaulter rate,
            by default None
        accumulated_defaulter_rate : bool, optional
            Accumulative defaulter rate (right to left), by default False
        buckets_as_quantiles : bool, optional
            Sort 10% of distribution for each bin, by default False
        n_buckets : int, optional
            Quantity of bins, by default 10., by default 10
        color_palette : str, optional
            Graph style, by default 'default'. You can choose between:
                - `default`: Default backtests color scheme
                - `scudra`: Scudra official color scheme
        """

        if np.count_nonzero(np.isnan(dist)):
            raise AttributeError(
                'The `dist` argument must not have NaN values!')
        if len(dist.shape) != 1:
            raise AttributeError('The `dist` argument must be an array 1-d!')
        if target is not None and len(target.shape) != 1:
            raise AttributeError('The `target` argument must be an array 1-d!')

        np.random.seed(24)  # ( ͡° ͜ʖ ͡°)
        dist_unique = dist + (np.random.rand(dist.shape[0]) / 1000)

        self.__palette = color_palette
        self.__breakpoints = self.calculate_breakpoints(
            dist_unique, n_buckets, buckets_as_quantiles)
        self.__dist_qtd, self.__dist_perc = self.calculate_distribution(
            dist_unique, self.__breakpoints)
        self.__defaulter_rate = self.calculate_defaulter_rate(
            dist_unique, target, self.__breakpoints,
            accumulated_defaulter_rate)
        self.make_title(buckets_as_quantiles, accumulated_defaulter_rate)

    def scale_range(self, input_values: np.array, min_value: float,
                    max_value: float) -> np.array:
        """Normalizes the `input_values` putting that given distribution
            between `min_value` and `max_value`.
            Parameters
            ----------
            input_values : np.array
                Raw distribution to be normalized.
            min_value : float
                Lower bound of the new distribution.
            max_value : float
                Upper bound of the new distribution.
            Returns
            -------
            np.array
                Returns the given array adjusted between the the lower and
                upper bound.
        """
        input_values = input_values.astype(float)
        input_values += -(np.min(input_values))
        input_values /= np.max(input_values) / (max_value - min_value)
        input_values += min_value
        return input_values

    def calculate_breakpoints(self, dist: np.array, n_bins: int,
                              buckets_as_quantiles: bool) -> np.array:
        """Calculates the boundaries of a distribution based on bucket type.
            Arranges the number of elements per bucket.
            Parameters
            ----------
            bucket_type : str
                bucket type (bins or quantiles).
                If 'bins', separates distribution based on fixed ranges.
                If 'quantiles', separates distribution based on percentiles.
            n_buckets : int
                Quantity of bins groups
            dist : np.array
                Distribution of interest.
            Returns
            -------
            np.array
                Breakpoints of distribution.
        """
        breakpoints = np.arange(0, n_bins + 1) / (n_bins) * 100

        if buckets_as_quantiles:
            breakpoints = np.stack(
                [np.percentile(dist, b) for b in breakpoints])
        else:
            breakpoints = np.array([i
                                    for i in np.arange(0., 1., 1. / n_bins)] +
                                   [1])

        return breakpoints

    def calculate_distribution(self, dist: np.array,
                               breakpoints: np.array) -> tuple:
        """Adjusts a given distribution according to the breakpoints.
            Parameters
            ----------
            dist : np.array
                Raw score distribution to be adjusted.
            breakpoints : np.array
                Breakpoints calculated by `calculate_breakpoints` method.
            Returns
            -------
            Tuple of lists. The first list refers to `dist` as well the
            second one refers to the `dist_b`.
                (list of amounts of scores per bucket,
                list of percentages of scores per bucket)
        """

        # Gets the quantity of scores for each bin
        dist_qtd = np.histogram(dist, breakpoints)[0]

        # Gets the population percentage of each bin
        dist_perc = dist_qtd / dist_qtd.shape[0]
        dist_perc[dist_perc == 0] = 0.0001
        dist_perc /= dist_perc.sum()

        return dist_qtd, dist_perc

    def accumulate_array(self, array: np.array) -> np.array:
        """Returns an array in which each item array[i] is of
        form array[i] = array[i] + array[i-1]

        Parameters
        ----------
        array : np.array
            Array within defaulter rate each bin calculated by
            `calculate_defaulter_rate` method.

        Returns
        -------
        np.array
            Accumulated defaulter rate (right to left)
        """
        return np.flip(np.cumsum(np.flip(array)))

    def calculate_defaulter_rate(self,
                                 dist: np.array,
                                 targets: np.array,
                                 breakpoints: np.array,
                                 accumulated=False) -> np.array:
        """Calculates the defaulter rate for each bin.

        Parameters
        ----------
        dist : np.array
            Distribution of scores whose defaulter rate must be calculated.
        targets : np.array
            Targets of the distribution. 0 means bad payer, 1 means good payer.
            A target value in index `i` refers to the score in the same index
            position in the score distribution.
        breakpoints : np.array
            Breakpoints of each bin calculated by `calculate_breakpoints`
            method.
        accumulated : bool, optional
            Calculates the defaulter rate using an accumulative approach
            implemented by `accumulate_array` method. By default False.

        Returns
        -------
        np.array
            defaulter rate for each bin in score distribution.
            Returns None if targets are not available.
        """

        if targets is None:
            return None

        if not accumulated:
            defaulter_rate = self.__calculate_standard_defaulter_rate(
                dist, targets, breakpoints)
            return defaulter_rate

        else:
            accum_defaulter_rate = self.__calculate_accum_defaulter_rate(
                dist, targets, breakpoints)
            return accum_defaulter_rate

    def __calculate_standard_defaulter_rate(self, dist: np.array,
                                            targets: np.array,
                                            breakpoints: np.array) -> np.array:
        """Calculates defaulter rate.

        Parameters
        ----------
        dist : np.array
            Distribution of scores whose defaulter rate must be calculated.
        targets : np.array
            Targets of the distribution. 0 means bad payer, 1 means good payer.
            A target value in index `i` refers to the score in the same index
            position in the score distribution.
        breakpoints : np.array
            Breakpoints of each bin calculated by `calculate_breakpoints`
            method.

        Returns
        -------
        np.array
            defaulter rate.
        """
        defaulter_rate = np.zeros(breakpoints.shape[0] - 1)

        for i in range(1, breakpoints.shape[0] - 1):
            bucket_index = ((dist >= breakpoints[i - 1])
                            & (dist < breakpoints[i]))
            defaulter_rate[i - 1] = 1 - targets[bucket_index].mean() if (
                len(targets[bucket_index]) > 0) else np.nan

        bucket_index = dist >= breakpoints[-2]
        defaulter_rate[-1] = 1 - targets[bucket_index].mean() if len(
            targets[bucket_index]) > 0 else 0
        defaulter_rate *= 100

        return np.nan_to_num(defaulter_rate)

    def __calculate_accum_defaulter_rate(self, dist: np.array,
                                         targets: np.array,
                                         breakpoints: np.array) -> np.array:
        """Calculates accumulated defaulter rate.

        Parameters
        ----------
        dist : np.array
            Distribution of scores whose defaulter rate must be calculated.
        targets : np.array
            Targets of the distribution. 0 means bad payer, 1 means good payer.
            A target value in index `i` refers to the score in the same index
            position in the score distribution.
        breakpoints : np.array
            Breakpoints of each bin calculated by `calculate_breakpoints`
            method.

        Returns
        -------
        np.array
            Accumulated defaulter rate.
        """
        global_default_rate = 1 - targets.sum() / targets.shape[0]
        debtors_per_range = np.zeros(breakpoints.shape[0] - 1)
        for i in range(1, breakpoints.shape[0] - 1):
            bucket_index = ((dist >= breakpoints[i - 1])
                            & (dist < breakpoints[i]))
            debtors_per_range[i - 1] = np.nan_to_num(
                (targets[bucket_index] == 0).sum())

        bucket_index = dist >= breakpoints[-2]
        debtors_per_range[-1] = np.nan_to_num(
            (targets[bucket_index] == 0).sum())

        defaulter_rate = 100 * debtors_per_range / sum(debtors_per_range)

        accum_defaulter_rate = global_default_rate * self.accumulate_array(
            defaulter_rate)
        return accum_defaulter_rate

    def make_title(self, bins_as_quantiles: bool,
                   accumulate_defaulter_rate: bool):
        """Creates a plot title based on object parameters"""

        title = '{}Score Distribution{}'.format(
            'Homogeneous ' if bins_as_quantiles else '',
            '\n(using accumulated delinquency rate)'
            if accumulate_defaulter_rate else '')

        self.__title = title

    def plot(self,
             title: str = None,
             ax: plt.axes = None,
             fontsize: int = 14,
             tight=False,
             figsize: tuple = None) -> None:
        """Make all the magic happens!! Plots the distribution adjusted
        according to the breakpoints. If targets are available it'll show the
        defaulter rate for each bin.
        ~ Vulture wings, chicken feathers, plot this cute graphic!! :D

        Parameters
        ----------
        title : str, optional
            Figure title, by default None.
            If None, then the class parameters gonna be used to generate
            an appropriated title.
        ax : plt.axes, optional
            Matplotlib Axis to export this output, by default None
        """

        if figsize == None:
            figsize = ((self.breakpoints.shape[0] - 1) * 1.5, 8)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize,
                                   facecolor=(1, 1, 1))

        title = self.plot_title if title is None else title
        x_axis = np.arange(0.1, self.breakpoints.shape[0] - 1 + .1)

        ax.set_title(title, {'size': fontsize if tight else (fontsize + 2)})
        ax.set_ylabel('Population (%)', {
            'size': fontsize,
            'weight': 'bold'
        },
                      color=PALETTES[self.palette]['primary_a'])
        ax.set_xlabel('Scores', {'size': fontsize})
        ax.set_xticks(x_axis)

        if tight or figsize != None:
            ax.set_xticklabels(
                [
                    "{:.0f}".format(100 * self.breakpoints[i])
                    for i in range(1, self.breakpoints.shape[0])
                ],
                {"size": fontsize - 3},
            )
        else:
            ax.set_xticklabels(
                [
                    "[{:.0f}, {:.0f})".format(100 * self.breakpoints[i - 1],
                                              100 * self.breakpoints[i])
                    for i in range(1, self.breakpoints.shape[0])
                ],
                {"size": fontsize},
            )

        ax.tick_params('y', colors=PALETTES[self.palette]['primary_a'])

        # Plots each bin
        ax.bar(x_axis,
               self.dist_perc * 100,
               0.68,
               color=PALETTES[self.palette]['primary_a'])

        # Gets the higher value of distribution (i.e., higher bin)
        higher_bin = np.max(self.dist_perc * 100)
        # Lambda function to get a percentage related to the higher bin
        per = lambda x: x / higher_bin

        # Plots the label for each bin in a appropriate position (y) using a
        # constractive font color
        for x, y in zip(x_axis, self.dist_perc):
            y *= 100
            label = y

            if per(y) <= .15:
                bar_text_color = PALETTES[self.palette]['primary_a']
                y += .8
            else:
                bar_text_color = COLORS['white']
                y -= 1. * (y * .2)

            ax.text(x,
                    y,
                    '%.2f' % label,
                    fontsize=(fontsize - 3) if tight else fontsize,
                    va='center',
                    ha='center',
                    color=bar_text_color)

        # If targets are available, plots the defaulter rate for each bin...
        if self.defaulter_rate is not None:
            ax2 = ax.twinx()

            if tight:
                ax2.get_yaxis().set_visible(False)
            else:
                ax2.set_ylabel('Delinquency rate  (%)', {
                    'size': fontsize,
                    'weight': 'bold'
                },
                               color=PALETTES[self.palette]['secondary_a'])

            ax2.set_ylim(0, 5 + self.defaulter_rate.max())
            ax2.tick_params('y', colors=PALETTES[self.palette]['secondary_a'])

            ax2.plot(x_axis,
                     self.defaulter_rate,
                     '-',
                     linewidth=3,
                     color=PALETTES[self.palette]['secondary_a'])

            for x, y in zip(x_axis, self.defaulter_rate):
                label = "{:.2f}".format(y)

                ax2.annotate(label, (x, y),
                             textcoords='data',
                             fontsize=fontsize - 2.5,
                             ha='center',
                             bbox=dict(
                                 boxstyle='round, pad=0.35',
                                 fc=PALETTES[self.palette]['secondary_b'],
                                 ec=PALETTES[self.palette]['secondary_b'],
                                 alpha=1))

    # Getters
    @property
    def palette(self):
        return self.__palette

    @property
    def plot_title(self):
        return self.__title

    @property
    def breakpoints(self):
        return self.__breakpoints

    @property
    def dist_qtd(self):
        return self.__dist_qtd

    @property
    def dist_perc(self):
        return self.__dist_perc

    @property
    def defaulter_rate(self):
        return self.__defaulter_rate

    @property
    def report(self):
        return {'Low Score Risk': self.value}
