# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import logging

logging.captureWarnings(True)

colors = ['b', 'orange', 'g', 'r', 'c', 'm',
          'y', 'k', 'Brown', 'ForestGreen'] * 30
# colors = ['orange', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen'] * 30
markers = ['+', 'x', 'p', '.', 'o', '8', 'p', '1', '*', '2', 'h'] * 30
plt.style.use('classic')


def exp_marginal(d, v0, sigma_v0):
    v_square = 0.5 * v0 ** 2 + sigma_v0 * d + 0.5 * \
        v0 * np.sqrt(v0 ** 2 + 4 * sigma_v0 * d)
    return np.exp(-d ** 2 / v_square)


v_exp_marginal = np.vectorize(exp_marginal)

class npcm_algorithm_fcm():
    def __init__(self, X, m, alpha_cut=0.1, error=0.001, maxiter=100):
        """
        :param X: scikit-learn form, i.e., pf shape (n_samples, n_features)
        :param m: NO.of initial clusters
        :param sig_v0:
        :return:
        """
        self.x = X
        self.m = m
        self.m_ori = m  # the original number of clusters specified
        self.alpha_cut = alpha_cut
        # alpha_cut can't be exactly 0, because we will use it to caculate
        # sig_vj via 0.2* ita/ sqrt(log(alpha_cut))
        if abs(self.alpha_cut) < 1e-5:
            self.alpha_cut += 1e-5
        # alpha_cut also shouldn't be too large, for the same reason as above
        if abs(1 - self.alpha_cut) < 1e-5:
            self.alpha_cut -= 1e-5
        self.error = error
        self.maxiter = maxiter
        self.log = logging.getLogger('algorithm.npcm')
        # use fcm to initialise the clusters
        self.init_theta_ita()
        pass
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

        # now compute ita
        ita = np.zeros(self.m)
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x - cntr[cntr_index])
            ita[cntr_index] = np.dot(dist_2_cntr, u_orig[:, cntr_index]) / sum(u_orig[:, cntr_index])
        self.ita = ita

        # clf = KMeans(self.m_ori, random_state=45).fit(self.x)
        # # hard classification labels
        # labels = clf.labels_
        # # initialize theta, i.e., the centers
        # self.theta = clf.cluster_centers_
        # # now compute ita
        # ita = np.zeros(self.m)
        # self.log.debug("Initialize bandwidth via KMeans")
        # for cntr_index in range(self.m_ori):
        #     dist_2_cntr = map(np.linalg.norm, self.x[
        #         labels == cntr_index] - self.theta[cntr_index])
        #     ita[cntr_index] = np.average(dist_2_cntr)
        #     self.log.debug("%d th cluster, ita:%3f" %
        #                    (cntr_index, ita[cntr_index]))
        # self.ita = ita

        # eliminate noise clusters
        density_list = []  # store density each cluster
        for index in range(self.m):
            no_of_pnts = np.sum(labels == index)
            # if np.isclose(ita[index], 0):
            # if there is only one point in the cluster, ita[index] would be
            # close to 0
            if no_of_pnts == 1:
                density = 0
            else:
                density = no_of_pnts / \
                          np.power(ita[index], np.shape(self.x)[1])
            density_list.append(density)
        index_delete = []  # store the cluster index to be deleted
        p = 0
        max_density = max(density_list)  # the maximum density
        for index in range(self.m):
            if density_list[index] < 0.1 * max_density:
                index_delete.append(index)
                p += 1
        for index in range(self.m):
            self.log.debug("%d th cluster, ita:%.3f, density:%.3f",
                           index, ita[index], density_list[index])
        self.log.debug("Noise cluster delete list:%s", index_delete)
        self.theta = np.delete(self.theta, index_delete, axis=0)
        self.ita = np.delete(self.ita, index_delete, axis=0)
        self.m -= p
        pass

    def update_u_theta(self):
        # update u (membership matrix)
        self.log.debug("Update parameters")
        self.ita_alpha_sigmaV = []
        self.ita_alpha_ori = []
        u = np.zeros((np.shape(self.x)[0], self.m))
        # tmp_sig=[]
        # for cntr_index in range(self.m):
        #     dist_2_cntr = map(np.linalg.norm, self.x - self.theta[cntr_index])
        #     tmp_sig_vj = 0.5 * self.ita[cntr_index] / np.sqrt(-np.log(self.alpha_cut))
        #     tmp_sig.append(tmp_sig_vj)
        # tmp_sig_vj=np.average(tmp_sig)
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x - self.theta[cntr_index])
            # caculate sigma_vj for each cluster
            tmp_sig_vj = 0.2 * self.ita[cntr_index] / \
                np.sqrt(-np.log(self.alpha_cut))
            # tmp_sig_vj = 1
            # caculate the d_\alpha of each cluster, i.e., the outer bandwidth
            # circle
            tmp_ita_alpha = np.sqrt(-np.log(self.alpha_cut)) * (self.ita[cntr_index] + np.sqrt(-np.log(self.alpha_cut))
                                                                * tmp_sig_vj)
            tmp_ita_ori = np.sqrt(-np.log(self.alpha_cut)
                                  ) * self.ita[cntr_index]
            self.ita_alpha_sigmaV.append(tmp_ita_alpha)
            self.ita_alpha_ori.append(tmp_ita_ori)
            self.log.debug("%d th cluster, ita:%3f, sig_v:%3f, d_alpha_corrected:%3f, d_alpha_ori:%3f" %
                           (cntr_index, self.ita[cntr_index], tmp_sig_vj, tmp_ita_alpha, tmp_ita_ori))
            u[:, cntr_index] = v_exp_marginal(
                dist_2_cntr, self.ita[cntr_index], tmp_sig_vj)
        self.u = u
        # update theta (centers)
        for cntr_index in range(self.m):
            # only those without too much noise can be used to calculate
            # centers
            samples_mask = u[:, cntr_index] >= self.alpha_cut
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
            self.ita_alpha_ori = np.delete(
                self.ita_alpha_ori, index_delete, axis=0)
            self.ita_alpha_sigmaV = np.delete(
                self.ita_alpha_sigmaV, index_delete, axis=0)
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
            # dist_2_cntr = map(np.linalg.norm, self.x[labels == cntr_index] - self.theta[cntr_index])
            # self.ita[cntr_index] = sum(dist_2_cntr) / np.sum(labels == cntr_index)
            # self.ita[cntr_index] = np.dot(dist_2_cntr, self.u[labels == cntr_index][:, cntr_index]) / np.sum(
            #     labels == cntr_index)
            samples_mask = np.logical_and(
                self.u[:, cntr_index] >= 0.01, labels == cntr_index)
            if np.any(samples_mask):
                dist_2_cntr = map(np.linalg.norm, self.x[
                                  samples_mask] - self.theta[cntr_index])
                self.ita[cntr_index] = sum(dist_2_cntr) / np.sum(samples_mask)
            else:
                self.ita[cntr_index] = 0

            # # only those without too much noise can be used to calculate ita
            # samples_mask = self.u[:, cntr_index] >= 0.1
            # # samples_mask = self.u[:, cntr_index] >= self.alpha_cut
            # tmp_average_mf=np.average(self.u[samples_mask][:, cntr_index])
            # self.log.info("cluster:%d,%3f",cntr_index,tmp_average_mf)
            # if np.any(samples_mask):  # avoid null value for the following calculation
            #     dist_2_cntr = map(np.linalg.norm, self.x[samples_mask] - self.theta[cntr_index])
            #     # self.ita[cntr_index] = np.dot(dist_2_cntr,self.u[samples_mask][:, cntr_index]) / np.sum(samples_mask)
            #     self.ita[cntr_index] = sum(dist_2_cntr)*tmp_average_mf / np.sum(samples_mask)
            if np.isclose(self.ita[cntr_index], 0):
                p += 1
                index_delete.append(cntr_index)
                # remove the respective center related quantities
        if p > 0:
            self.u = np.delete(self.u, index_delete, axis=1)
            self.theta = np.delete(self.theta, index_delete, axis=0)
            self.ita = np.delete(self.ita, index_delete, axis=0)
            self.ita_alpha_ori = np.delete(
                self.ita_alpha_ori, index_delete, axis=0)
            self.ita_alpha_sigmaV = np.delete(
                self.ita_alpha_sigmaV, index_delete, axis=0)
            self.m -= p
        self.labels = np.argmax(self.u, axis=1)
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
                tmp < self.error )
                # print "fraction:", tmp
                if flag:
                    # self.save_last_frame(p)
                    self.p=p
                    break
            except:
                pass

            theta_pre = theta_ori.copy()
            p += 1
            # here the current iteration result has been recorded in the class,
            # the result is ready for plotting.
            # yield p
            # note that the yield statement returns p as an argument to the callback function __call__(self, p) which is called by the
            # animation process
        return self
class npcm_algorithm():
    def __init__(self, X, m, alpha_cut=0.1, error=0.001, maxiter=100):
        """
        :param X: scikit-learn form, i.e., pf shape (n_samples, n_features)
        :param m: NO.of initial clusters
        :param sig_v0:
        :return:
        """
        self.x = X
        self.m = m
        self.m_ori = m  # the original number of clusters specified
        self.alpha_cut = alpha_cut
        # alpha_cut can't be exactly 0, because we will use it to caculate
        # sig_vj via 0.2* ita/ sqrt(log(alpha_cut))
        if abs(self.alpha_cut) < 1e-5:
            self.alpha_cut += 1e-5
        # alpha_cut also shouldn't be too large, for the same reason as above
        if abs(1 - self.alpha_cut) < 1e-5:
            self.alpha_cut -= 1e-5
        self.error = error
        self.maxiter = maxiter
        self.log = logging.getLogger('algorithm.npcm')
        # use fcm to initialise the clusters
        self.init_theta_ita()
        pass
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
        # x_tmp = self.x.T  # becasue skfuzzy is converted from matlab, the data should be of (n_features, n_samples)
        # cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(x_tmp, self.m, 2, error=0.005, maxiter=1000, seed=45)
        # # cntr, u_orig represent center,fuzzy c-partitioned matrix (membership matrix)
        # u_orig = u_orig.T  # convert back to scikit-learn form (n_samples, n_features)
        # # I'm confused that the returned cntr is already (n_samples, n_features) and doesn't need the transpose
        # # plot the fcm initialization
        # labels = np.argmax(u_orig, axis=1)
        # # initialize theta, i.e., the centers
        # self.theta = cntr
        #
        # # now compute ita
        # ita = np.zeros(self.m)
        # for cntr_index in range(self.m):
        #     dist_2_cntr = map(np.linalg.norm, self.x - cntr[cntr_index])
        #     ita[cntr_index] = np.dot(dist_2_cntr, u_orig[:, cntr_index]) / sum(u_orig[:, cntr_index])
        # self.ita = ita

        clf = KMeans(self.m_ori, random_state=45).fit(self.x)
        # hard classification labels
        labels = clf.labels_
        # initialize theta, i.e., the centers
        self.theta = clf.cluster_centers_
        # now compute ita
        ita = np.zeros(self.m)
        self.log.debug("Initialize bandwidth via KMeans")
        for cntr_index in range(self.m_ori):
            dist_2_cntr = map(np.linalg.norm, self.x[
                labels == cntr_index] - self.theta[cntr_index])
            ita[cntr_index] = np.average(dist_2_cntr)
            self.log.debug("%d th cluster, ita:%3f" %
                           (cntr_index, ita[cntr_index]))
        self.ita = ita

        # eliminate noise clusters
        density_list = []  # store density each cluster
        for index in range(self.m):
            no_of_pnts = np.sum(labels == index)
            # if np.isclose(ita[index], 0):
            # if there is only one point in the cluster, ita[index] would be
            # close to 0
            if no_of_pnts == 1:
                density = 0
            else:
                density = no_of_pnts / \
                          np.power(ita[index], np.shape(self.x)[1])
            density_list.append(density)
        index_delete = []  # store the cluster index to be deleted
        p = 0
        max_density = max(density_list)  # the maximum density
        for index in range(self.m):
            if density_list[index] < 0.1 * max_density:
                index_delete.append(index)
                p += 1
        for index in range(self.m):
            self.log.debug("%d th cluster, ita:%.3f, density:%.3f",
                           index, ita[index], density_list[index])
        self.log.debug("Noise cluster delete list:%s", index_delete)
        self.theta = np.delete(self.theta, index_delete, axis=0)
        self.ita = np.delete(self.ita, index_delete, axis=0)
        self.m -= p
        pass

    def update_u_theta(self):
        # update u (membership matrix)
        self.log.debug("Update parameters")
        self.ita_alpha_sigmaV = []
        self.ita_alpha_ori = []
        u = np.zeros((np.shape(self.x)[0], self.m))
        # tmp_sig=[]
        # for cntr_index in range(self.m):
        #     dist_2_cntr = map(np.linalg.norm, self.x - self.theta[cntr_index])
        #     tmp_sig_vj = 0.5 * self.ita[cntr_index] / np.sqrt(-np.log(self.alpha_cut))
        #     tmp_sig.append(tmp_sig_vj)
        # tmp_sig_vj=np.average(tmp_sig)
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x - self.theta[cntr_index])
            # caculate sigma_vj for each cluster
            tmp_sig_vj = 0.2 * self.ita[cntr_index] / \
                np.sqrt(-np.log(self.alpha_cut))
            # tmp_sig_vj = 1
            # caculate the d_\alpha of each cluster, i.e., the outer bandwidth
            # circle
            tmp_ita_alpha = np.sqrt(-np.log(self.alpha_cut)) * (self.ita[cntr_index] + np.sqrt(-np.log(self.alpha_cut))
                                                                * tmp_sig_vj)
            tmp_ita_ori = np.sqrt(-np.log(self.alpha_cut)
                                  ) * self.ita[cntr_index]
            self.ita_alpha_sigmaV.append(tmp_ita_alpha)
            self.ita_alpha_ori.append(tmp_ita_ori)
            self.log.debug("%d th cluster, ita:%3f, sig_v:%3f, d_alpha_corrected:%3f, d_alpha_ori:%3f" %
                           (cntr_index, self.ita[cntr_index], tmp_sig_vj, tmp_ita_alpha, tmp_ita_ori))
            u[:, cntr_index] = v_exp_marginal(
                dist_2_cntr, self.ita[cntr_index], tmp_sig_vj)
        self.u = u
        # update theta (centers)
        for cntr_index in range(self.m):
            # only those without too much noise can be used to calculate
            # centers
            samples_mask = u[:, cntr_index] >= self.alpha_cut
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
            self.ita_alpha_ori = np.delete(
                self.ita_alpha_ori, index_delete, axis=0)
            self.ita_alpha_sigmaV = np.delete(
                self.ita_alpha_sigmaV, index_delete, axis=0)
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
            # dist_2_cntr = map(np.linalg.norm, self.x[labels == cntr_index] - self.theta[cntr_index])
            # self.ita[cntr_index] = sum(dist_2_cntr) / np.sum(labels == cntr_index)
            # self.ita[cntr_index] = np.dot(dist_2_cntr, self.u[labels == cntr_index][:, cntr_index]) / np.sum(
            #     labels == cntr_index)
            samples_mask = np.logical_and(
                self.u[:, cntr_index] >= 0.01, labels == cntr_index)
            if np.any(samples_mask):
                dist_2_cntr = map(np.linalg.norm, self.x[
                                  samples_mask] - self.theta[cntr_index])
                self.ita[cntr_index] = sum(dist_2_cntr) / np.sum(samples_mask)
            else:
                self.ita[cntr_index] = 0

            # # only those without too much noise can be used to calculate ita
            # samples_mask = self.u[:, cntr_index] >= 0.1
            # # samples_mask = self.u[:, cntr_index] >= self.alpha_cut
            # tmp_average_mf=np.average(self.u[samples_mask][:, cntr_index])
            # self.log.info("cluster:%d,%3f",cntr_index,tmp_average_mf)
            # if np.any(samples_mask):  # avoid null value for the following calculation
            #     dist_2_cntr = map(np.linalg.norm, self.x[samples_mask] - self.theta[cntr_index])
            #     # self.ita[cntr_index] = np.dot(dist_2_cntr,self.u[samples_mask][:, cntr_index]) / np.sum(samples_mask)
            #     self.ita[cntr_index] = sum(dist_2_cntr)*tmp_average_mf / np.sum(samples_mask)
            if np.isclose(self.ita[cntr_index], 0):
                p += 1
                index_delete.append(cntr_index)
                # remove the respective center related quantities
        if p > 0:
            self.u = np.delete(self.u, index_delete, axis=1)
            self.theta = np.delete(self.theta, index_delete, axis=0)
            self.ita = np.delete(self.ita, index_delete, axis=0)
            self.ita_alpha_ori = np.delete(
                self.ita_alpha_ori, index_delete, axis=0)
            self.ita_alpha_sigmaV = np.delete(
                self.ita_alpha_sigmaV, index_delete, axis=0)
            self.m -= p
        self.labels = np.argmax(self.u, axis=1)
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
                tmp < self.error )
                # print "fraction:", tmp
                if flag:
                    # self.save_last_frame(p)
                    self.p = p
                    break
            except:
                pass

            theta_pre = theta_ori.copy()
            p += 1
            self.p = p
            # here the current iteration result has been recorded in the class,
            # the result is ready for plotting.
            # yield p
            # note that the yield statement returns p as an argument to the callback function __call__(self, p) which is called by the
            # animation process
        return self

class upcm_algorithm():
    def __init__(self, X, m, sig_v0,alpha_cut=0.1, error=0.001, maxiter=1000):
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
        self.alpha_cut = alpha_cut
        self.error = error
        self.maxiter = maxiter
        # use fcm to initialise the clusters
        self.init_theta_ita()
        pass

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
                    tmp < self.error)
                # print "fraction:", tmp
                if flag:
                    self.m_ = self.m # final cluster number
                    self.p = p
                    break
            except:
                pass
            p += 1
            # here the current iteration result has been recorded in the class,
            # the result is ready for plotting.
            # yield p
            # note that the yield statement returns p as an argument to the callback function __call__(self, p) which is called by the
            # animation process
            self.p=p
        return self


class pcm_algorithm():
    def __init__(self, X, m,  error=0.001, maxiter=100):
        """
        :param X: scikit-learn form, i.e., pf shape (n_samples, n_features)
        :param m: NO.of initial clusters
        :param alpha:
        :return:
        """
        self.x = X
        self.m = m
        self.m_ori = m  # the original number of clusters specified
        self.error = error
        self.maxiter = maxiter
        # use fcm to initialise the clusters
        self.init_theta_ita()
        pass

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
            ita[cntr_index] = np.dot(np.square(dist_2_cntr), u_orig[:, cntr_index]) / sum(u_orig[:, cntr_index])
        self.ita = ita
        self.ita_hat = min(ita)
        pass

    def update_u_theta(self):
        # update u (membership matrix)
        u = np.zeros((np.shape(self.x)[0], self.m))
        for cntr_index in range(self.m):
            dist_2_cntr = map(np.linalg.norm, self.x - self.theta[cntr_index])
            new_ita = 1 / self.ita[cntr_index]  # just caculated as usual
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

    def fit(self):
        # The main loop
        p = 0
        while p < self.maxiter:
            theta_ori = self.theta.copy()
            try:
                labels_ori = np.argmax(self.u, axis=1)
            except:
                pass
            self.update_u_theta()
            self.cluster_elimination()
            self.labels = np.argmax(self.u, axis=1)
            # self.adapt_ita()
            try:
                tmp = float(np.count_nonzero(self.labels != labels_ori)) / len(self.labels)
                flag = (len(theta_ori) == len(self.theta)) and (
                    tmp < self.error)
                # print "fraction:", tmp
                if flag:
                    self.p = p
                    break
            except:
                pass

            theta_pre = theta_ori.copy()
            p += 1
            # here the current iteration result has been recorded in the class,
            # the result is ready for plotting.
            # yield p
            # note that the yield statement returns p as an argument to the callback function __call__(self, p) which is called by the
            # animation process
        return self


class apcm_algorithm():
    def __init__(self, X, m, alpha, error=0.001, maxiter=100):
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
        self.error = error
        self.maxiter = maxiter
        # use fcm to initialise the clusters
        self.init_theta_ita()
        pass
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
                    tmp < self.error)
                # print "fraction:", tmp
                if flag:
                    self.p = p
                    break
            except:
                pass

            theta_pre = theta_ori.copy()
            p += 1
            # here the current iteration result has been recorded in the class,
            # the result is ready for plotting.
            # yield p
            # note that the yield statement returns p as an argument to the callback function __call__(self, p) which is called by the
            # animation process
        return self

