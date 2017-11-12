# -*- coding: utf-8 -*-
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import matplotlib.animation as animation

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
plt.style.use('ggplot')


class apcm():
    def __init__(self, X, m, alpha,ax, x_lim, y_lim, error=0.001, maxiter=10000):
        """
        :param X: scikit-learn form, i.e., pf shape (n_samples, n_features)
        :param m: NO.of initial clusters
        :param alpha:
        :return:
        """
        self.x = X
        self.m = m
        self.m_ori = m  # the original number of clusters specified
        self.alpha = float(alpha)
        self.ax=ax
        self.x_lim=x_lim#tuple
        self.y_lim=y_lim
        self.error = error
        self.maxiter = maxiter
        # use fcm to initialise the clusters
        self.init_theta_ita()
        pass

    def init_animation(self):
        ax=self.ax
        # initialise the lines to update (each line represents a cluster)
        # this idea comes from http://stackoverflow.com/questions/19519587/python-matplotlib-plot-multi-lines-array-and-animation
        # i.e.,o animate N lines, you just need N Line2D objects
        self.lines = [ax.plot([], [], '.', color=colors[label])[0] for label in range(self.m_ori)]
        # centers
        # self.line_centers=[ax.plot([],[], 's',color=colors[label])[0] for label in range(self.m_ori) ]
        self.line_centers = [ax.plot([], [], 'rs')[0] for _ in range(self.m_ori)]
        # the title
        self.title = ax.set_title("")
        # the limit of axixes
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        # add circles to indication the standard deviation, i.e., the influence of each cluster
        self.circles=[ax.add_patch(plt.Circle((0,0),radius=0,color='k',fill=None,lw=2)) for _ in range(self.m_ori)]
        return self.lines,self.line_centers,self.title,self.circles

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

        # plot the fcm initialization result
        # fig, ax = plt.subplots()# assume 2-d data
        # for label in range(self.m):
        #     ax.plot(self.x[labels == label][:,0], self.x[labels == label][:,1], '.',
        #          color=colors[label])
        # ax.set_title('the fcm initialization')

        # initialize theta, i.e., the centers
        self.theta = cntr
        # now compute ita
        ita = np.zeros(self.m)
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x - cntr[cntr_index])
            ita[cntr_index] = np.dot(dist_2_cntr, u_orig[:, cntr_index]) / sum(u_orig[:, cntr_index])
        self.ita = ita
        self.ita_hat = min(ita)
        # print "In init:self.ita_hat/self.alpha",self.ita_hat/self.alpha
        pass

    def update_u_theta(self):
        # update u (membership matrix)
        u = np.zeros((np.shape(self.x)[0], self.m))
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x - self.theta[cntr_index])
            new_ita = self.alpha / (self.ita_hat * self.ita[cntr_index])
            u[:, cntr_index] = np.exp(-new_ita * np.square(dist_2_cntr))
        self.u = u
        # update theta (centers)
        for cntr_index in range(self.m):
            self.theta[cntr_index] = np.sum(u[:, cntr_index][:, np.newaxis] * self.x, axis=0) / sum(u[:, cntr_index])
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
        p=0
        index_delete = []  # store the cluster index to be deleted

        labels = np.argmax(self.u, axis=1)
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x[labels == cntr_index] - self.theta[cntr_index])
            self.ita[cntr_index] = sum(dist_2_cntr) / np.sum(labels == cntr_index)
            if np.isclose(self.ita[cntr_index],0):
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

    def fit(self):
        # The main loop
        p = 0
        while p < self.maxiter:
            theta_ori = self.theta.copy()
            self.update_u_theta()
            self.cluster_elimination()
            self.adapt_ita()

            p += 1
            if (len(theta_ori) == len(self.theta)) and (np.sum(np.absolute(self.theta - theta_ori)) < self.error):
                print 'Final cluster number:',len(self.theta)
                break
            yield p  # here the current iteration result has been recorded in the class, the result is ready for plotting.
            # note that the yield statement returns p as an argument to the callback function __call__(self, p) which is called by the
            # animation process

    def __call__(self, p):
        """
        (refer to 74.4 animation example code: bayes_update.py from Matplotlib, Release 1.4.3 page1632)
        :param p:
        :return:
        """
        self.title.set_text("%d th iteration." % p)
        # self.title.set_text(p)
        labels = np.argmax(self.u, axis=1)
        # the following logic is as this: if the final cluster number is equal to the specified value
        #then draw all the clusters, otherwise, the deleted clusters are not plotted
        if self.m==self.m_ori:
            for label,line,line_center,circle in zip(range(self.m),self.lines,self.line_centers,self.circles):
                line.set_data(self.x[labels==label][:,0],self.x[labels==label][:,1])
                line_center.set_data(self.theta[label][0],self.theta[label][1])
                circle.center=self.theta[label][0],self.theta[label][1]
                circle.set_radius(self.ita[label])
        else:
            for label,line ,line_center,circle in zip(range(self.m),self.lines[:self.m],self.line_centers[:self.m],self.circles[:self.m]):
                line.set_data(self.x[labels==label][:,0],self.x[labels==label][:,1])
                line_center.set_data(self.theta[label][0],self.theta[label][1])
                circle.center=self.theta[label][0],self.theta[label][1]
                circle.set_radius(self.ita[label])
                # print label,self.ita[label]
            for label,line ,line_center,circle in zip(range(self.m,self.m_ori),self.lines[self.m:],self.line_centers[self.m:],self.circles[self.m:]):
                line.set_data([],[])
                line_center.set_data([],[])
                circle.set_radius(0)
        # return self.lines,self.line_centers,self.title
        return self.lines,self.line_centers,self.title,self.circles
