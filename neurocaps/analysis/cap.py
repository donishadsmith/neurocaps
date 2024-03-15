from typing import Union
from sklearn.cluster import KMeans
from kneed import KneeLocator
from .._utils import _CAPGetter, _convert_pickle_to_dict
import numpy as np, re, warnings

class CAP(_CAPGetter):
    def __init__(self, parcel_approach: dict[dict], n_clusters: Union[int, list[int]]=5, cluster_selection_method: str=None, groups: dict=None):
        """
        Initialize the CAP (Co-activation Patterns) analysis class.

        Parameters
        ----------
        node_labels : list[str]
            Decoded or non-decoded Schaefer Atlas labels for the nodes.
        n_clusters : int or list[int], default=5
            The number of clusters to use. Can be a single integer or a list of integers.
        cluster_selection_method: str, default=None
            Method to find the optimal number of clusters. Options are "silhouette" or "elbow".
        groups : dict, default=None
            A mapping of group names to subject IDs. Each group contains subject IDs for
            separate CAP analysis. If None, CAPs are not separated by group.

        Raises
        ------
        ValueError
            If `cluster_selection_method` is none when `n_clusters` is a list.
        ValueError
            If `cluster_selection_method` is used when `n_clusters` is a single integer.
        TypeError
            If `groups` is provided but is not a dictionary.
        AssertionError
            If any group in `groups` has zero subject IDs.

        Notes
        -----
        The initialization ensures unique values if `n_clusters` is a list and checks for 
        valid input types and values.
        """
        # Ensure all unique values if n_clusters is a list
        self._n_clusters = n_clusters if type(n_clusters) == int else sorted(list(set(n_clusters)))
        self._cluster_selection_method = cluster_selection_method 

        if type(n_clusters) == list:
            self._n_clusters =  self._n_clusters[0] if all([type(self._n_clusters) == list, len(self._n_clusters) == 1]) else self._n_clusters
            # Raise error if n_clusters is a list and no cluster selection method is specified
            if all([len(n_clusters) > 1, self._cluster_selection_method== None]):
                raise ValueError("`cluster_selection_method` cannot be None since n_clusters is a list.")

        # Raise error if silhouette_method is requested when n_clusters is an integer
        if all([self._cluster_selection_method != None, type(self._n_clusters) == int]):
            raise ValueError("`cluster_selection_method` only valid if n_clusters is a range of unique integers.")
       
        self._groups = groups
        # Raise error if self groups is not a dictionary 
        if self._groups:
            if type(self._groups) != dict:
                raise TypeError("`groups` must be a dictionary where the keys are the group names and the items correspond to subject ids in the groups.")
            
            for group_name in self._groups.keys():
                assert len(self._groups[group_name]) > 0, f"{group_name} has zero subject ids."
            
            # Convert ids to strings
            for group in self._groups.keys():
                self._groups[group] = [str(subj_id) if not isinstance(subj_id,str) else subj_id for subj_id in self._groups[group]]
        
        valid_parcel_dict = {"Schaefer": {"n_rois" : 400, "yeo_networks": 7},
                         "AAL": {"version": "SPM12"}}

        if len(parcel_approach.keys()) > 1:
            raise ValueError(f"Only one parcellation approach can be selected from the following valid options: {valid_parcel_dict.keys()}.\n Example format of `parcel_approach`: {valid_parcel_dict}")
        
        self._parcel_approach = parcel_approach 

    def get_caps(self, subject_timeseries: Union[dict[dict[np.ndarray]], str], runs: Union[int, list[int]]=None, random_state: int=None, show_figs: bool=True, standardize: bool=True, epsilon: Union[int,float]=0, **kwargs) -> None:
        """"" Create CAPs

        The purpose of this function is to concatenate the timeseries of each subject and perform kmeans clustering on the concatenated data.
        
        Parameters
        ----------
        subject_timeseries: dict or str
            A pickle file containing the nested subject timeseries dictionary saved by the TimeSeriesExtractor class or
            the nested subject timeseries dictionary produced by the TimeseriesExtractor class. The first level of the nested dictionary must consist of the subject
            ID as a string, the second level must consist of the the run numbers in the form of 'run-#', where # is the corresponding number of the run, and the last level 
            must consist of the timeseries associated with that run.
        runs: int or list[int], default=None
            The run numbers to perform the CAPs analysis with. If None, all runs in the subject timeseries will be concatenated into a single dataframe.
        random_state: int, default=None
            The random state to use for scikits KMeans function.
        show_figs: bool, default=True
            Display the plots of inertia scores for all groups if `cluster_selection_method`="elbow".
        standardize: bool, default=True
            To z-score the features of the concatonated timeseries array.
        epsilon: int or float, default=0
            Small number to add to the denominator when z-scoring for numerical stabilty.
        kwargs: dect
            Dictionary to adjust the sensitivity, `S` parameter, of the elbow method. The elbow method uses the KneeLocator function from the  kneed package. If no `S` is inputted, `S` will be KneeLocator default.
            Larger values of `S` are more conservative and less sensitive to small fluctuations.
            
        Raises
        ------
        ValueError
            If both input_path and subject_timeseries are None.
        """
        
        if runs:
            if isinstance(runs,int): runs = list(runs)
    
        self._runs = runs if runs else "all"
        self._standardize = standardize
        self._epsilon = epsilon

        if isinstance(subject_timeseries, str) and "pkl" in subject_timeseries: subject_timeseries = _convert_pickle_to_dict(pickle_file=subject_timeseries)

        self._concatenated_timeseries = self._get_concatenated_timeseries(subject_timeseries=subject_timeseries, runs=runs)

        if self._cluster_selection_method == "silhouette": 
            self._perform_silhouette_method(random_state=random_state)
        elif self._cluster_selection_method == "elbow":
            self._perform_elbow_method(random_state=random_state, show_figs=show_figs, **kwargs)
        else:
            self._kmeans = {}
            for group in self._groups.keys():
                self._kmeans[group] = {}
                self._kmeans[group] = KMeans(n_clusters=self._n_clusters,random_state=random_state).fit(self._concatenated_timeseries[group]) if random_state or random_state == 0 else KMeans(n_clusters=self._n_clusters).fit(self._concatenated_timeseries[group])
            
        # Create states dict   
        self._create_caps_dict()
    
    def _perform_silhouette_method(self, random_state):
        from sklearn.metrics import silhouette_score

        # Initialize attribute
        self._silhouette_scores = {}
        self._optimal_n_clusters = {}
        self._kmeans = {}

        for group in self._groups.keys():
            self._silhouette_scores[group] = {}
            for n_cluster in self._n_clusters:
                self._kmeans[group] = KMeans(n_clusters=n_cluster,random_state=random_state).fit(self._concatenated_timeseries[group]) if random_state or random_state == 0 else KMeans(n_clusters=n_cluster).fit(self._concatenated_timeseries[group])
                cluster_labels = self._kmeans[group].predict(self._concatenated_timeseries[group])
                self._silhouette_scores[group].update({n_cluster: silhouette_score(self._concatenated_timeseries[group], cluster_labels)})
            self._optimal_n_clusters[group] = max(self._silhouette_scores[group], key=self._silhouette_scores[group].get)
            if self._optimal_n_clusters[group] != self._n_clusters[-1]:
                self._kmeans[group] = KMeans(n_clusters=self._optimal_n_clusters[group],random_state=random_state).fit(self._concatenated_timeseries[group]) if random_state or random_state == 0 else KMeans(n_clusters=self._optimal_n_clusters[group]).fit(self._concatenated_timeseries[group])
            print(f"Optimal cluster size for {group} is {self._optimal_n_clusters[group]}.")
        
    
    def _perform_elbow_method(self, random_state, show_figs, **kwargs):
        # Initialize attribute
        self._inertia = {}
        self._optimal_n_clusters = {}
        self._kmeans = {}

        knee_dict = dict(S = kwargs["S"] if "S" in kwargs.keys() else 1)

        for group in self._groups.keys():
            self._inertia[group] = {}
            for n_cluster in self._n_clusters:
                self._kmeans[group] = KMeans(n_clusters=n_cluster,random_state=random_state).fit(self._concatenated_timeseries[group]) if random_state or random_state == 0 else KMeans(n_clusters=n_cluster).fit(self._concatenated_timeseries[group])
                self._inertia[group].update({n_cluster: self._kmeans[group].inertia_}) 
            
            # Get optimal cluster size
            kneedle = KneeLocator(x=list(self._inertia[group].keys()), 
                                                        y=list(self._inertia[group].values()),
                                                        curve='convex',
                                                        direction='decreasing', S=knee_dict["S"])
                
            self._optimal_n_clusters[group] = kneedle.elbow
            if not self._optimal_n_clusters[group]:
                 warnings.warn("No elbow detected so optimal cluster size is None. Try adjusting the sensitivity parameter, `S`, to increase or decrease sensitivity (higher values are less sensitive), expanding the list of clusters to test, or setting `cluster_selection_method` to 'sillhouette'.")
            else:
                if self._optimal_n_clusters[group] != self._n_clusters[-1]:
                    self._kmeans[group] = KMeans(n_clusters=self._optimal_n_clusters[group],random_state=random_state).fit(self._concatenated_timeseries[group]) if random_state or random_state == 0 else KMeans(n_clusters=self._optimal_n_clusters[group]).fit(self._concatenated_timeseries[group])
                print(f"Optimal cluster size for {group} is {self._optimal_n_clusters[group]}.\n")

                if show_figs:
                    kneedle.plot_knee(title=group)
                    kneedle.plot_knee_normalized(title=group)

    def _create_caps_dict(self):
        # Initialize dictionary
        self._caps = {}
        for group in self._groups.keys():
                self._caps[group] = {}
                self._caps[group].update({f"CAP-{state_number}": state_vector for state_number, state_vector in zip([num for num in range(1,len(self._kmeans[group].cluster_centers_)+1)],self._kmeans[group].cluster_centers_)})
    
    def _get_concatenated_timeseries(self, subject_timeseries, runs):
        # Create dictionary for "All Subjects" if no groups are specified to reuse the same loop instead of having to create logic for grouped and non-grouped version of the same code
        if not self._groups: self._groups = {"All Subjects": [subject for subject in subject_timeseries.keys()]}

        concatenated_timeseries = {group: {} for group in self._groups.keys()}

        self._generate_lookup_table()

        self._mean_vec = {group: {} for group in self._groups.keys()}
        self._stdev_vec = {group: {} for group in self._groups.keys()}

        for subj_id, group in self._subject_table.items():
            requested_runs = [f"run-{run}" for run in runs] if runs else subject_timeseries[subj_id].keys()
            subject_runs = [subject_run for subject_run in subject_timeseries[subj_id].keys() if subject_run in requested_runs] 
            if len(subject_runs) == 0:
                warnings.warn(f"Skipping subject {subj_id} since they do not have the requested run numbers {','.join(requested_runs)}")
                continue
            for curr_run in subject_runs:
                if len(concatenated_timeseries[group]) == 0:
                    concatenated_timeseries[group] = subject_timeseries[subj_id][curr_run] if subj_id in list(set(self._groups[group])) else concatenated_timeseries[group]
                else:
                    concatenated_timeseries[group] = np.vstack([concatenated_timeseries[group], subject_timeseries[subj_id][curr_run]]) if subj_id in list(set(self._groups[group])) else concatenated_timeseries[group]
        # Standardize
        if self._standardize:
            for group in self._groups.keys():
                self._mean_vec[group], self._stdev_vec[group] = np.mean(concatenated_timeseries[group], axis=0), np.std(concatenated_timeseries[group], ddof=1, axis=0)
                concatenated_timeseries[group] = (concatenated_timeseries[group] - self._mean_vec[group])/(self._stdev_vec[group] + self._epsilon)

        return concatenated_timeseries
    
    def _generate_lookup_table(self):
        self._subject_table = {}
        for group in self._groups:
            for subj_id in self._groups[group]:
                if subj_id in self._subject_table.keys():
                    warnings.warn(f"Subject: {subj_id} appears more than once, only including the first instance of this subject in the analysis.") 
                else:
                    self._subject_table.update({subj_id : group})

    def visualize_caps(self, output_dir: str=None, plot_options: Union[str, list[str]]="outer product", visual_scope: list[str]="networks", task_title: str=None, show_figs: bool=True, subplots: bool=False, **kwargs):
        """ Plotting CAPs

        This function produces seaborn heatmaps for each CAP. If groups were given when the CAP class was initialized, plotting will be done for all CAPs for all groups.

        Parameters
        ----------
        output_dir: str, default=None
            Directory to save plots in. If None, plots will not be saved.
        plot_options: str or list[str], default="outer product"
            Type of plots to create. Options are "outer product" or "heatmap".
        visual_scope: str or list[str], default="networks
            Determines whether plotting is done at the network level or node level. For network level, the value of each nodes in the same networks are averaged together them plotted.
        task_title: str, default=None
            Serves as the title of each plot as well as the name of the saved file if output_dir is given.
        show_figs: bool, default=True
            Display figures or not to display figures.
        subplots: bool, default=True
            Produce subplots for outer product plots.
        kwargs: dict
            Keyword arguments used when saving figures. Valid keywords include "dpi", "format", "figsize", "fontsize", "hspace", "wspace", "xticklabels_size", "yticklabels_size", "shrink", "nrow", "ncol", "suptitle_fontsize", "tight_layout", "rect", "sharey", "xlabel_rotation", "ylabel_rotation", "annot". 
            If `output_dir` is not None and no inputs for dpi and format are given, dpi defaults to 300 and format defaults to "png". If no keywords, "figsize" defaults to (8,6), "fontsize", which adjusts the title size of the individual plots or subplots, defaults to 14, "hspace", which adjusts spacing for subplots, defaults to 0.4, 
            "wspace", which adjusts spacing between subplots, defaults to 0.4, "xticklabels_size" defaults to 8, "yticklabels_size" defaults to 8, shrink, which adjusts the cbar size, defaults to 0.8, "nrow", which is the number of rows for subplot and varies, and "ncol", which is the number of columns for subplot, default varies but max is 5, "suptitle_fontsize", 
            size of the main title when subplot is True, defaults to 0.7, "tight_layout", use tight layout for subplot, defaults to True, "rect", input for the `rect` parameter in tight layout when subplots is True to fix whitespace issues, default is [0, 0.03, 1, 0.95], "sharey", which shares y axis labela for subplots, defaults to True, 
            "xlabel_rotation", which rotates the labels on the x-axis, defaults to 0, "ylabel_rotation", which rotates the labels on the y-axis, defaults to 0, "annot", which adds values to cells on the outer product heatmap at the network level only, defaults to False .
    
        """
        import os, itertools

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # Convert to list
        if type(plot_options) == str: plot_options = [plot_options]
        if type(visual_scope) == str: visual_scope = [visual_scope]

        # Check inputs for plot_options and visual_scope
        if not any(["heatmap" in plot_options, "outer product" in plot_options]):
            raise ValueError("Valid inputs for `plot_options` are 'heatmap' and 'outer product'.")
        
        if not any(["networks" in visual_scope, "nodes" in visual_scope]):
            raise ValueError("Valid inputs for `visual_scope` are 'networks' and 'nodes'.")

        if "networks" in visual_scope: self._create_networks()

        # Create plot dictionary
        plot_dict = dict(dpi = kwargs["dpi"] if kwargs and "dpi" in kwargs.keys() else 300,
                        format = kwargs["format"] if kwargs and "format" in kwargs.keys() else "png",
                        figsize = kwargs["figsize"] if kwargs and "figsize" in kwargs.keys() else (8,6),
                        fontsize = kwargs["fontsize"] if kwargs and "fontsize" in kwargs.keys() else 14,
                        hspace = kwargs["hspace"] if kwargs and "hspace" in kwargs.keys() else 0.2,
                        wspace = kwargs["wspace"] if kwargs and "wspace" in kwargs.keys() else 0.2,
                        xticklabels_size = kwargs["xticklabels_size"] if kwargs and "xticklabels_size" in kwargs.keys() else 8,
                        yticklabels_size = kwargs["yticklabels_size"] if kwargs and "yticklabels_size" in kwargs.keys() else 8,
                        shrink = kwargs["shrink"] if kwargs and "shrink" in kwargs.keys() else 0.8,
                        nrow = kwargs["nrow"] if kwargs and "nrow" in kwargs.keys() else None,
                        ncol = kwargs["ncol"] if kwargs and "ncol" in kwargs.keys() else None,
                        suptitle_fontsize = kwargs["suptitle_fontsize"] if kwargs and "suptitle_fontsize" in kwargs.keys() else 20,
                        tight_layout = kwargs["tight_layout"] if kwargs and "tight_layout" in kwargs.keys() else True,
                        rect = kwargs["rect"] if kwargs and "rect" in kwargs.keys() else [0, 0.03, 1, 0.95],
                        sharey = kwargs["sharey"] if kwargs and "sharey" in kwargs.keys() else True,
                        xlabel_rotation = kwargs["xlabel_rotation"] if kwargs and "xlabel_rotation" in kwargs.keys() else 0,
                        ylabel_rotation = kwargs["ylabel_rotation"] if kwargs and "ylabel_rotation" in kwargs.keys() else 0,
                        annot = kwargs["annot"] if kwargs and "annot" in kwargs.keys() else False,
                        )
        
        if kwargs:
            invalid_kwargs = {key : value for key, value in kwargs.items() if key not in plot_dict.keys()}
            if len(invalid_kwargs.keys()) > 0:
                print(f"Invalid kwargs arguments used and will be ignored {invalid_kwargs}.")

        # Check labels
        check_caps = self._caps[list(self._caps.keys())[0]]
        check_caps = check_caps[list(check_caps.keys())[0]]
        if check_caps.shape[0] != len(self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]): 
                raise ValueError("Number of rois/labels used for CAPs does not equal the number of rois/labels specified in `parcel_approach`.")
                
        # Ensure plot_options and visual_scope are lists
        plot_options = plot_options if type(plot_options) == list else list(plot_options)
        visual_scope = visual_scope if type(visual_scope) == list else list(visual_scope)
        # Initialize outer product attribute
        if "outer product" in plot_options: self._outer_product = {}

        distributed_list = list(itertools.product(plot_options,visual_scope,self._groups.keys()))

        for plot_option, scope, group in distributed_list:
                # Get correct labels depending on scope
                if scope == "networks": cap_dict, columns = self._network_caps, self._parcel_approach[list(self._parcel_approach.keys())[0]]["networks"]
                elif scope == "nodes": cap_dict, columns = self._caps, self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]

                #  Generate plot for each group
                if plot_option == "outer product": self._generate_outer_product_plots(group=group, plot_dict=plot_dict, cap_dict=cap_dict, columns=columns, subplots=subplots,
                                                                                    output_dir=output_dir, task_title=task_title, show_figs=show_figs, scope=scope)
                elif plot_option == "heatmap": self._generate_heatmap_plots(group=group, plot_dict=plot_dict, cap_dict=cap_dict, columns=columns,
                                                                            output_dir=output_dir, task_title=task_title, show_figs=show_figs, scope=scope)
            
    def _create_networks(self):
        self._network_caps = {group: {} for group in self._groups.keys()}
        for group in self._groups.keys():
            for cap in self._caps[group].keys():
                network_caps = {}
                for network in self._parcel_approach[list(self._parcel_approach.keys())[0]]["networks"]:
                    if len(network_caps) == 0:
                        network_caps = np.array([np.average(self._caps[group][cap][np.array([index for index, node in enumerate(self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]) if network in node])])])
                    else:
                        network_caps = np.hstack([network_caps, np.average(self._caps[group][cap][np.array([index for index, node in enumerate(self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]) if network in node])])])
            
                self._network_caps[group].update({cap: network_caps})
    
    def _generate_outer_product_plots(self, group, plot_dict, cap_dict, columns, subplots, output_dir, task_title, show_figs, scope):
        from seaborn import heatmap
        import matplotlib.pyplot as plt, os

        # Nested dictionary for group
        self._outer_product[group] = {}

        # Create base grid for subplots
        if subplots:
            # Max five subplots per row for default
            default_col = len(cap_dict[group].keys()) if len(cap_dict[group].keys()) <= 5 else 5
            ncol = plot_dict["ncol"] if plot_dict["ncol"] != None else default_col
            # Pad nrow, since int will round down, padding is needed for cases where len(cap_dict[group].keys())/ncol is a float. This will add the extra row needed
            x_pad = 0 if len(cap_dict[group].keys())/ncol <= 1 else 1
            nrow = plot_dict["nrow"] if plot_dict["nrow"] != None else x_pad + int(len(cap_dict[group].keys())/ncol)

            subplot_figsize = (8 * ncol, 6 * nrow) if plot_dict["figsize"] == (8,6) else plot_dict["figsize"] 

            fig, axes = plt.subplots(nrow, ncol, sharex=False, sharey=plot_dict["sharey"], figsize=subplot_figsize)
            suptitle = f"{group} {task_title}" if task_title else f"{group}"
            fig.suptitle(suptitle, fontsize=plot_dict["suptitle_fontsize"])
            fig.subplots_adjust(hspace=plot_dict["hspace"], wspace=plot_dict["wspace"])  
            if plot_dict["tight_layout"]: fig.tight_layout(rect=plot_dict["rect"])  

            # Current subplot
            axes_x, axes_y = [0,0] 

        # Iterate over CAPs
        for cap in cap_dict[group].keys():
            # Calculate outer product
            self._outer_product[group].update({cap: np.outer(cap_dict[group][cap],cap_dict[group][cap])})
            if subplots: 
                ax = axes[axes_y] if nrow == 1 else axes[axes_x,axes_y]
                # Modify tick labels based on scope
                if scope == "networks":
                    display = heatmap(ax=ax, data=self._outer_product[group][cap], cmap="coolwarm", xticklabels=columns, yticklabels=columns, cbar_kws={"shrink": plot_dict["shrink"]}, annot=plot_dict["annot"])
                else:
                    # Create Labels
                    import collections
                    
                    if list(self._parcel_approach.keys())[0] == "Schaefer":
                        frequency_dict = dict(collections.Counter([names[0] + " " + names[1] for names in [name.split("_")[0:2] for name in self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]]]))
                    elif list(self._parcel_approach.keys())[0] == "AAL":
                        frequency_dict = collections.Counter([name.split("_")[0] for name in self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]])
                    names_list = list(frequency_dict.keys())
                    labels = ["" for _ in range(0,len(self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]))]

                    shift = 0

                    for num, name in enumerate(names_list):
                        if num == 0:
                            labels[0] = name
                        else:
                            shift += frequency_dict[names_list[num-1]] 
                            indx = (frequency_dict[names_list[num]]) + shift - frequency_dict[names_list[num]]
                            labels[indx] = name

                    display = heatmap(ax=ax, data=self._outer_product[group][cap], cmap="coolwarm", cbar_kws={"shrink": plot_dict["shrink"]})

                    ticks = [i for i, label in enumerate(labels) if label]  

                    ax.set_xticks(ticks)  
                    ax.set_xticklabels([label for label in labels if label]) 
                    ax.set_yticks(ticks)  
                    ax.set_yticklabels([label for label in labels if label]) 

                # Modify label sizes
                display.set_xticklabels(display.get_xticklabels(), size = plot_dict["xticklabels_size"], rotation=plot_dict["xlabel_rotation"])

                if plot_dict["sharey"] == True:
                    if axes_y == 0: display.set_yticklabels(display.get_yticklabels(), size = plot_dict["yticklabels_size"], rotation=plot_dict["ylabel_rotation"])
                else:
                    display.set_yticklabels(display.get_yticklabels(), size = plot_dict["yticklabels_size"], rotation=plot_dict["ylabel_rotation"])

                # Set title of subplot
                ax.set_title(cap, fontsize=plot_dict["fontsize"])

                # If modulus is zero, move onto the new column back to zero
                if (axes_y % ncol == 0 and axes_y != 0) or axes_y == ncol - 1: 
                    axes_x += 1
                    axes_y = 0
                else:
                    axes_y += 1

            else:
                # Create new plot for each iteration when not subplot
                plt.figure(figsize=plot_dict["figsize"])

                plot_title = f"{group} {cap} {task_title}" if task_title else f"{group} {cap}"
                if scope == "networks": display = heatmap(self._outer_product[group][cap], cmap="coolwarm", xticklabels=columns, yticklabels=columns, cbar_kws={'shrink': plot_dict["shrink"]})
                else: display = heatmap(self._outer_product[group][cap], cmap="coolwarm", xticklabels=[], yticklabels=[], cbar_kws={'shrink': plot_dict["shrink"]})
                
                # Set title
                display.set_title(plot_title, fontdict= {'fontsize': plot_dict["fontsize"]})

                # Modify label sizes
                display.set_xticklabels(display.get_xticklabels(), size = plot_dict["xticklabels_size"], rotation=plot_dict["xlabel_rotation"])
                display.set_yticklabels(display.get_yticklabels(), size = plot_dict["yticklabels_size"], rotation=plot_dict["ylabel_rotation"])

                # Save individual plots
                if output_dir:
                    partial_filename = f"{group}_{cap}_{task_title}" if task_title else f"{group}_{cap}"
                    full_filename = f"{partial_filename}_outer_product_heatmap-networks.{plot_dict['format']}" if scope == "networks" else f"{partial_filename}_outer_product_heatmap-nodes.{plot_dict['format']}"
                    display.get_figure().savefig(os.path.join(output_dir,full_filename), dpi=plot_dict["dpi"], bbox_inches='tight')
        
        # Remove subplots with no data
        if subplots: [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]

        # Save subplot
        if subplots and output_dir: 
            partial_filename = f"{group}_CAPS_{task_title}" if task_title else f"{group}_CAPS"
            full_filename = f"{partial_filename}_outer_product_heatmap-networks.{plot_dict['format']}" if scope == "networks" else f"{partial_filename}_outer_product_heatmap-nodes.{plot_dict['format']}"
            display.get_figure().savefig(os.path.join(output_dir,full_filename), dpi=plot_dict["dpi"], bbox_inches='tight')    
        
        # Display figures
        if not show_figs: plt.close()

    def _generate_heatmap_plots(self, group, plot_dict, cap_dict, columns, output_dir, task_title, show_figs, scope):
        from seaborn import heatmap
        import matplotlib.pyplot as plt, pandas as pd, os
        
        # Initialize new grid
        plt.figure(figsize=plot_dict["figsize"])

        if scope == "networks": display = heatmap(pd.DataFrame(cap_dict[group], index=columns), cmap='coolwarm', cbar_kws={'shrink': plot_dict["shrink"]}) 
        else: 
            # Create Labels
            import collections
            if list(self._parcel_approach.keys())[0] == "Schaefer":
                frequency_dict = dict(collections.Counter([names[0] + " " + names[1] for names in [name.split("_")[0:2] for name in self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]]]))
            elif list(self._parcel_approach.keys())[0] == "AAL":
                frequency_dict = collections.Counter([name.split("_")[0] for name in self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]])
            names_list = list(frequency_dict.keys())
            labels = ["" for _ in range(0,len(self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]))]

            shift = 0

            for num, name in enumerate(names_list):
                if num == 0:
                    labels[0] = name
                else:
                    shift += frequency_dict[names_list[num-1]] 
                    indx = (frequency_dict[names_list[num]]) + shift - frequency_dict[names_list[num]]
                    labels[indx] = name

            display = heatmap(pd.DataFrame(cap_dict[group], columns=cap_dict[group].keys()), cmap='coolwarm', cbar_kws={'shrink': plot_dict["shrink"]})

            plt.yticks(ticks=[pos for pos, label in enumerate(labels) if label], labels=names_list)  

        # Modify label sizes
        display.set_xticklabels(display.get_xticklabels(), size = plot_dict["xticklabels_size"], rotation=plot_dict["xlabel_rotation"])
        display.set_yticklabels(display.get_yticklabels(), size = plot_dict["yticklabels_size"], rotation=plot_dict["ylabel_rotation"])

        # Set plot name
        plot_title = f"{group} CAPS {task_title}" if task_title else f"{group} CAPS" 
        display.set_title(plot_title, fontdict= {'fontsize': plot_dict["fontsize"]})

        # Save plots
        if output_dir:
            partial_filename = f"{group}_CAPS_{task_title}" if task_title else f"{group}_CAPS"
            full_filename = f"{partial_filename}_heatmap-networks.{plot_dict['format']}" if scope == "networks" else f"{partial_filename}_heatmap-nodes.{plot_dict['format']}"
            display.get_figure().savefig(os.path.join(output_dir,full_filename), dpi=plot_dict["dpi"], bbox_inches='tight')    
   
        # Display figures
        if not show_figs: plt.close()

    def calculate_metrics(self, subject_timeseries: Union[dict[dict[np.ndarray]], str], tr: float=None, runs: Union[int]=None, continuous_runs: bool=False, metrics: Union[str, list[str]]=["temporal fraction", "persistence", "counts", "transition frequency"], return_df: bool=True, output_dir: str=None, file_name: str=None) -> dict:
        """Get CAP metrics

        Creates a single pandas Dataframe containing all participants containing CAP metrics as described in Liu et al., 2018 and Yang et al., 2021, where `temporal fraction` is the proportion of total volumes spent in a single CAP over all volumes in a run,
        `persistence` is the average time spent in a single CAP before transitioning to another CAP (average consecutive/uninterrupted time), and `counts` is the frequency of each CAP observed in a run, and `transition frequency` is the number of switches between
        different CAPs across the entire run.

        Parameters
        ----------
        subject_timeseries: dict or str
            The absolute path of the pickle file containing the nested subject timeseries dictionary saved by the TimeSeriesExtractor class or
            the nested subject timeseries dictionary produced by the TimeseriesExtractor class. The first level of the nested dictionary must consist of the subject
            ID as a string, the second level must consist of the the run numbers in the form of 'run-#', where # is the corresponding number of the run, and the last level 
            must consist of the timeseries associated with that run. This should be the same dictionary used for `get_caps()`.
        tr: float, default=None
            The repetition time. If given, persistence will be calculate as the average uninterrupted time spent in each CAP. If not give, persistence will be calculate
            as the average uninterrupted volumes (TRs) spent in each state.
        runs: int or list[int], default=None
            The run numbers to calculate cap metrics for. If None, cap metrics will be calculated for each run.
        continuous_runs: bool, default=False
            If True, all runs will be treated as a single, uninterrupted run.
        metrics : str or list[str], default=["temporal fraction", "persistence", "counts", "transition frequency"]
            The metrics to calculate. Available options include `temporal fraction`, `persistence`, `counts`, and `transition frequency`.
        return_df: str, default=True
            If True, dataframe will be returned.
        output_dir: str, default=None
            Directory to save dataframe in. If None, dataframe will not be saved.
        file_name: str, default=None
            Name to save dataframe as if output_dir is not None.

        Returns
        -------
        dictionary containing pandas dataframes - one for each metric requested.

        Raises
        ------
        ValueError
            No valid metrics are detected in `metrics` parameter.

        Notes
        -----
        The presence of NaN values for specific CAPs in the "temporal fraction", "persistence", or "counts" dataframes, indicates that the participant had zero instances of 
        a specific CAP.

        References
        -----
        Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

        Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. 
        NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

        """
        import collections, pandas as pd, os

        metrics = [metrics] if isinstance(metrics, str) else metrics

        valid_metrics = ["temporal fraction", "persistence", "counts", "transition frequency"]

        boolean_list = [element in valid_metrics for element in metrics]

        if any(boolean_list):
            invalid_metrics = [metrics[indx] for indx,boolean in enumerate(boolean_list) if boolean == False]
            if len(invalid_metrics) > 0:
                warnings.warn(f"invalid metrics will be ignored: {' '.join(invalid_metrics)}")
        else:
            raise ValueError(f"No valid metrics in `metrics` list. Valid metrics are {', '.join(valid_metrics)}")
        
        if isinstance(subject_timeseries, str) and "pkl" in subject_timeseries: subject_timeseries = _convert_pickle_to_dict(pickle_file=subject_timeseries)

        group_cap_dict = {}
        # Get group with most CAPs
        for group in self._groups.keys():
            group_cap_dict.update({group: len(self._caps[group])})
        
        cap_names =  self._caps[max(group_cap_dict, key=group_cap_dict.get)].keys()
        cap_numbers = [int(name.split("-")[-1]) for name in cap_names]

        # Assign each subject TRs to CAP
        predicted_subject_timeseries = {}

        for subj_id, group in self._subject_table.items():
            predicted_subject_timeseries[subj_id] = {}
            requested_runs = [f"run-{run}" for run in runs] if runs else subject_timeseries[subj_id].keys()
            subject_runs = [subject_run for subject_run in subject_timeseries[subj_id].keys() if subject_run in requested_runs] 
            if len(subject_runs) == 0:
                warnings.warn(f"Skipping subject {subj_id} since they do not have the requested run numbers {','.join(requested_runs)}")
                continue
            if not continuous_runs or len(requested_runs) == 1:
                for curr_run in subject_runs:
                        timeseries = (subject_timeseries[subj_id][curr_run] - self._mean_vec[group])/(self._stdev_vec[group] + self._epsilon) if self._standardize else subject_timeseries[subj_id][curr_run] 
                        predicted_subject_timeseries[subj_id].update({curr_run: self._kmeans[group].predict(timeseries) + 1})
            else:
                subject_runs = "Continuous Runs"
                timeseries = {subject_runs: {}}
                for curr_run in subject_timeseries[subj_id].keys():
                    timeseries[subject_runs] = np.vstack([timeseries[subject_runs], subject_timeseries[subj_id][curr_run]]) if len(timeseries[subject_runs]) != 0 else subject_timeseries[subj_id][curr_run]
                timeseries = (timeseries[subject_runs] - self._mean_vec[group])/(self._stdev_vec[group] + self._epsilon) if self._standardize else timeseries[subject_runs]
                predicted_subject_timeseries[subj_id].update({subject_runs: self._kmeans[group].predict(timeseries) + 1})

        # Create pd.dataframe
        df_dict = {}

        for metric in metrics: 
            if metric in valid_metrics:
                if metric != "transition frequency":
                    df_dict.update({metric: pd.DataFrame(columns=["Subject_ID", "Group","Run"] + list(cap_names))})
                else:
                    df_dict.update({metric: pd.DataFrame(columns=["Subject_ID", "Group","Run","Transition_Frequency"])})

        distributed_list = []
        for subj_id, group in self._subject_table.items():
            for curr_run in predicted_subject_timeseries[subj_id].keys():
                distributed_list.append([subj_id,group,curr_run])

        for subj_id, group, curr_run in distributed_list:
            if "temporal fraction" in metrics or "counts" in metrics:
                frequency_dict = dict(collections.Counter(predicted_subject_timeseries[subj_id][curr_run]))
                sorted_frequency_dict = {key: frequency_dict[key] for key in sorted(list(frequency_dict.keys()))}
                if len(sorted_frequency_dict) != len(cap_numbers):
                    sorted_frequency_dict = {cap_number: sorted_frequency_dict[cap_number] if cap_number in sorted_frequency_dict.keys() else float("nan") for cap_number in cap_numbers}
                if "temporal fraction" in metrics: 
                    proportion_dict = {key: item/(len(predicted_subject_timeseries[subj_id][curr_run])) for key, item in sorted_frequency_dict.items()}
                    # Populate Dataframe
                    df_dict["temporal fraction"].loc[len(df_dict["temporal fraction"])] = [subj_id, group, curr_run] + [items for _ , items in proportion_dict.items()]
                if "counts" in metrics:
                    # Populate Dataframe
                    df_dict["counts"].loc[len(df_dict["counts"])] = [subj_id, group, curr_run] + [items for _ , items in sorted_frequency_dict.items()]
            if "persistence" in metrics:
                # Initialize variable
                persistence_dict = {}
                uninterrupted_volumes = []
                count = 0

                # Iterate through caps
                for target in cap_numbers:
                    # Iterate through each element and count uniterrupted volumes that equal target
                    for index in range(0,len(predicted_subject_timeseries[subj_id][curr_run])):
                        if predicted_subject_timeseries[subj_id][curr_run][index] == target:
                            count +=1
                        # Store count in list if interuptted and not zero an
                        else:
                            if count != 0:
                                uninterrupted_volumes.append(count)
                            # Reset counter 
                            count = 0
                    # In the event, a participant only occupies one CAP and to ensure final counts are added
                    if count > 0:
                        uninterrupted_volumes.append(count)
                    # If uninterrupted_volumes not zero, multiply elements in the list by repetition time, sum and divide
                    if len(uninterrupted_volumes) > 0:
                        if tr:
                            persistence_dict.update({target: np.sum(np.array(uninterrupted_volumes)*tr)/(len(uninterrupted_volumes))})
                        else:
                            persistence_dict.update({target: np.sum(np.array(uninterrupted_volumes))/(len(uninterrupted_volumes))})
                    else:
                        persistence_dict.update({target: float("nan")})
                    # Reset variables
                    count = 0
                    uninterrupted_volumes = []
                # Populate Dataframe
                df_dict["persistence"].loc[len(df_dict["persistence"])] = [subj_id, group, curr_run] + [items for _ , items in persistence_dict.items()]
            if "transition frequency" in metrics:
                count = 0
                # Iterate through predicted values 
                for index in range(0,len(predicted_subject_timeseries[subj_id][curr_run])):
                    if index != 0:
                        # If the subsequent element does not equal the previous element, this is considered a transition
                        if predicted_subject_timeseries[subj_id][curr_run][index-1] != predicted_subject_timeseries[subj_id][curr_run][index]:
                            count +=1
                df_dict["transition frequency"].loc[len(df_dict["transition frequency"])] = [subj_id, group, curr_run, count]

        if output_dir:
            for metric in df_dict.keys():
                filename = file_name + f"-{metric.replace(' ','_')}" if file_name else f"{metric.replace(' ','_')}"
                df_dict[f"{metric}"].to_csv(path_or_buf=os.path.join(output_dir,filename + ".csv"), sep=",", index=False)

        if return_df:
            return df_dict




                    




