# -*- coding: utf-8 -*-
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import matplotlib.animation as animation

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen'] * 3
markers = ['+', 'x', 'p', '.', 'o', '8', 'p', '1', '*', '2', 'h'] * 30
plt.style.use('classic')


def exp_marginal(d, v0, sigma_v0):
    v_square = 0.5 * v0 ** 2 + sigma_v0 * d + 0.5 * v0 * np.sqrt(v0 ** 2 + 4 * sigma_v0 * d)
    return np.exp(-d ** 2 / v_square)


v_exp_marginal = np.vectorize(exp_marginal)


class upcm():
    def __init__(self, X, m, sig_v0, ax, x_lim, y_lim, alpha_cut=0.1, error=1e-5, maxiter=10000, ini_save_name="",
                 last_frame_name=""):
        """
        :param X: scikit-learn form, i.e., pf shape (n_samples, n_features)
        :param m: NO.of initial clusters
        :param sig_v0:
        :return:
        """
        self.x = X
        self.m = m
        self.m_ori = m  # the original number of clusters specified
        self.sig_v0 = float(sig_v0)
        self.ax = ax
        self.x_lim = x_lim  # tuple
        self.y_lim = y_lim
        self.alpha_cut = alpha_cut
        self.error = error
        self.maxiter = maxiter
        self.ini_save_name = ini_save_name
        self.last_frame_name = last_frame_name
        # use fcm to initialise the clusters
        self.init_theta_ita()
        pass

    def init_animation(self):
        ax = self.ax
        # initialise the lines to update (each line represents a cluster)
        # this idea comes from http://stackoverflow.com/questions/19519587/python-matplotlib-plot-multi-lines-array-and-animation
        # i.e.,o animate N lines, you just need N Line2D objects
        self.lines = [ax.plot([], [], linestyle='None', marker=markers[label], color=colors[label])[0] for label in
                      range(self.m_ori)]
        # centers
        # self.line_centers=[ax.plot([],[], 's',color=colors[label])[0] for label in range(self.m_ori) ]
        self.line_centers = [ax.plot([], [], 'rs')[0] for _ in range(self.m_ori)]
        # the title
        self.text = ax.text(0.02, 0.84, '', transform=ax.transAxes)
        # the limit of axixes
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        # add circles to indication the standard deviation, i.e., the influence of each cluster
        self.circles = [ax.add_patch(plt.Circle((0, 0), radius=0, color='k', fill=None, lw=2)) for _ in
                        range(self.m_ori)]
        return self.lines + self.line_centers + self.circles + [self.text]

    def init_theta_ita(self):
        """
        This initialization is criticle because pcm based algorithms all rely on this initial 'beautiful' placement
        of the cluster prototypes.

        As we know, pcm is good at mode-seeking, and the algorithm works quite intuitively: you specify the location
        of lots of prototypes, then the algorithm tries to seek the dense region around the prototypes.

        Note that the prototypes has very little probability to move far from there initial locations. This fact
        quit annoys me because it reveals the mystery secret of clustering and makes clustering a trival work. This
        fact also makes clustering unattractive any more.

        Recall that the motivation of pcm is to remove the strong constraint imposed on the memberships as in fcm, and this modification
        does have very good results, that is, the resulting memberships finally have the interpretation of typicality
        which is one of the most commonly used interpretations of memberships in applications of fuzzy set theory,
        and beacuse of this improvment, the algorithm behaves more wisely under noisy environment.

        I start to doubt the foundations of the clustering community.

        :return:
        """
        x_tmp = self.x.T  # becasue skfuzzy is converted from matlab, the data should be of (n_features, n_samples)
        cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(x_tmp, self.m, 2, error=0.005, maxiter=1000, seed=45)
        # cntr, u_orig represent center,fuzzy c-partitioned matrix (membership matrix)
        u_orig = u_orig.T  # convert back to scikit-learn form (n_samples, n_features)
        # I'm confused that the returned cntr is already (n_samples, n_features) and doesn't need the transpose
        # plot the fcm initialization
        labels = np.argmax(u_orig, axis=1)
        # initialize theta, i.e., the centers
        self.theta = cntr

        # plot the fcm initialization result
        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)  # assume 2-d data
        bbox_props = dict(boxstyle="circle,pad=0.1", fc='w', ec="k", lw=2, alpha=0.7)
        for label in range(self.m):
            ax.plot(self.x[labels == label][:, 0], self.x[labels == label][:, 1], linestyle='None',
                    marker=markers[label], color=colors[label])
            ax.text(self.theta[label][0], self.theta[label][1], "%d" % label, size='medium', ha="center", va="center",
                    bbox=bbox_props)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.grid(True)
        # ax.set_title('FCM initialization:%2d clusters' % self.m)
        plt.savefig(self.ini_save_name, dpi=fig.dpi, bbox_inches='tight')

        # now compute ita
        ita = np.zeros(self.m)
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x - cntr[cntr_index])
            ita[cntr_index] = np.dot(dist_2_cntr, u_orig[:, cntr_index]) / sum(u_orig[:, cntr_index])
        self.ita = ita
        pass

    def update_u_theta(self):
        # update u (membership matrix)
        u = np.zeros((np.shape(self.x)[0], self.m))
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x - self.theta[cntr_index])
            u[:, cntr_index] = v_exp_marginal(dist_2_cntr, self.ita[cntr_index], self.sig_v0)
        self.u = u
        # update theta (centers)
        for cntr_index in range(self.m):
            samples_mask = u[:,
                           cntr_index] >= self.alpha_cut  # only those without too much noise can be used to calculate centers
            if np.any(samples_mask):  # avoid null value for the following calculation
                self.theta[cntr_index] = np.sum(u[samples_mask][:, cntr_index][:, np.newaxis]
                                                * self.x[samples_mask], axis=0) / sum(u[samples_mask][:, cntr_index])
        pass

    def cluster_elimination(self):
        labels = np.argmax(self.u, axis=1)
        p = 0
        index_delete = []  # store the cluster index to be deleted
        for cntr_index in range(self.m):
            if np.any(labels == cntr_index):
                continue
            else:
                p += 1
                index_delete.append(cntr_index)
        # remove the respective center related quantities
        if p > 0:
            self.u = np.delete(self.u, index_delete, axis=1)
            self.theta = np.delete(self.theta, index_delete, axis=0)
            self.ita = np.delete(self.ita, index_delete, axis=0)
            self.m -= p
        pass

    def adapt_ita(self):
        """
        in the hard partition, if no point belongs to cluster i then it will be removed.
        :return:
        """
        p = 0
        index_delete = []  # store the cluster index to be deleted

        labels = np.argmax(self.u, axis=1)
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x[labels == cntr_index] - self.theta[cntr_index])
            self.ita[cntr_index] = sum(dist_2_cntr) / np.sum(labels == cntr_index)
            if np.isclose(self.ita[cntr_index], 0):
                p += 1
                index_delete.append(cntr_index)
                # remove the respective center related quantities
        if p > 0:
            self.u = np.delete(self.u, index_delete, axis=1)
            self.theta = np.delete(self.theta, index_delete, axis=0)
            self.ita = np.delete(self.ita, index_delete, axis=0)
            self.m -= p
        self.labels = np.argmax(self.u, axis=1)
        pass

    def save_last_frame(self, p):
        fig = plt.figure("last frame", dpi=300, figsize=(7, 3.5))
        ax = fig.gca()
        ax.grid(True)
        # the limit of axixes
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        # tmp_text = "Iteration times:%2d\n" % p
        # tmp_text += r"$\alpha={:.2f},\sigma_v={:.2f}$".format(self.alpha_cut, self.sig_v0) + "\n"
        # tmp_text += "Initial    number:%2d\nCurrent number:%2d" % (self.m_ori, self.m)
        tmp_text = "Initial    number:%2d\nCurrent number:%2d" % (self.m_ori, self.m)
        ax.text(0.02, 0.86, tmp_text, transform=ax.transAxes)
        # ax.set_title("Clustering Finished")
        labels = np.argmax(self.u, axis=1)
        for label in range(self.m):
            ax.plot(self.x[labels == label][:, 0], self.x[labels == label][:, 1], linestyle='None',
                    marker=markers[label], color=colors[label], zorder=1)
            ax.plot(self.theta[label][0], self.theta[label][1], 'rs', zorder=2)
            ax.add_patch(plt.Circle((self.theta[label][0], self.theta[label][1]), zorder=3,
                                    radius=self.ita[label], color='k', fill=None, lw=2))
        plt.figure("last frame")
        plt.savefig(self.last_frame_name, dpi=300, bbox_inches='tight')
        pass

    def fit(self):
        """
         # This re-initialization is necessary if we use animation.save. The reason is: FuncAnimation needs a
        # save_count parameter to know the  mount of frame data to keep around for saving movies. So the animation
        # first runs the fit() function to get the number of runs of the algorithm and save the movie, then this number
        #  is  the run times for the next animation run. This second run is the one we see, not the one we save.
        #  So we should make sure that the second run of fit() has exactly the same enviroment as the first run.
        :return:
        """
        # The main loop
        p = 0
        self.m = self.m_ori
        self.init_animation()
        self.init_theta_ita()
        while p < self.maxiter:
            theta_ori = self.theta.copy()
            try:
                labels_ori = np.argmax(self.u, axis=1)
            except:
                pass
            self.update_u_theta()
            self.cluster_elimination()
            self.adapt_ita()
            try:
                tmp = float(np.count_nonzero(self.labels != labels_ori)) / len(self.labels)
                flag = (len(theta_ori) == len(self.theta)) and (
                    tmp < 0.001)
                print "fraction:", tmp
                if flag:
                    self.save_last_frame(p)
                    break
            except:
                pass

            theta_pre = theta_ori.copy()
            p += 1
            # here the current iteration result has been recorded in the class,
            # the result is ready for plotting.
            yield p
            # note that the yield statement returns p as an argument to the callback function __call__(self, p) which is called by the
            # animation process

    def __call__(self, p):
        """
        (refer to 74.4 animation example code: bayes_update.py from Matplotlib, Release 1.4.3 page1632)
        :param p:
        :return:
        """
        print "in call:", p
        tmp_text = "Iteration times:%2d\n" % p
        tmp_text += r"$\alpha={:.2f},\sigma_v={:.2f}$".format(self.alpha_cut, self.sig_v0) + "\n"
        labels = np.argmax(self.u, axis=1)
        # the following logic is as this: if the final cluster number is equal to the specified value
        # then draw all the clusters, otherwise, the deleted clusters are not plotted
        if self.m == self.m_ori:
            for label, line, line_center, circle in zip(range(self.m), self.lines, self.line_centers, self.circles):
                line.set_data(self.x[labels == label][:, 0], self.x[labels == label][:, 1])
                line_center.set_data(self.theta[label][0], self.theta[label][1])
                circle.center = self.theta[label][0], self.theta[label][1]
                circle.set_radius(self.ita[label])
                # print label, self.ita[label], len(self.ita)
        else:
            for label, line, line_center, circle in zip(range(self.m), self.lines[:self.m], self.line_centers[:self.m],
                                                        self.circles[:self.m]):
                line.set_data(self.x[labels == label][:, 0], self.x[labels == label][:, 1])
                line_center.set_data(self.theta[label][0], self.theta[label][1])
                circle.center = self.theta[label][0], self.theta[label][1]
                circle.set_radius(self.ita[label])
                print label, self.ita[label], len(self.ita)
            for label, line, line_center, circle in zip(range(self.m, self.m_ori), self.lines[self.m:],
                                                        self.line_centers[self.m:], self.circles[self.m:]):
                line.set_data([], [])
                line_center.set_data([], [])
                circle.set_radius(0)
        tmp_text += "Initial    number:%2d\nCurrent number:%2d" % (self.m_ori, self.m)
        self.text.set_text(tmp_text)
        return self.lines + self.line_centers + self.circles + [self.text]
