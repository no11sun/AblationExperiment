class SACAgent:
    def __init__(self,args,envs,reward_):
        self.n_node = args.n_node
        self.xdim = args.n_node * args.n_node
        self.pol = make_mlp([self.xdim] +
                                  [args.n_hid] * args.n_layers + [self.xdim], tail =[nn.Softmax(1)])

        self.Q_1 = make_mlp([self.xdim] +
                                  [args.n_hid] * args.n_layers + [self.xdim])
        self.Q_2 = make_mlp([self.xdim] +
                            [args.n_hid] * args.n_layers + [self.xdim])
        self.Q_t1 = make_mlp([self.xdim] +
                            [args.n_hid] * args.n_layers + [self.xdim])
        self.Q_t2 = make_mlp([self.xdim] +
                            [args.n_hid] * args.n_layers + [self.xdim])
        self.reward_  = reward_
        self.alpha = torch.tensor([args.sac_alpha],requires_grad = True)
        self.alpha_target = args.sac_alpha
        self.reward_ = reward_
        self.envs = envs
        self.tau = args.bootstrap_tau

    def parameters(self):
        return ( list(self.pol.parameters()) + list(self.Q_1.parameters()) +
                 list(self.Q_2.parameters()) + [self.alpha])

    def sample_many(self,mbsize,s, masked_list):
        # batch = []
        d = args.n_node
        masked_list_new = copy.deepcopy(masked_list)
        done = [False] * mbsize
        trajs = defaultdict(list)
        transitive_list = copy.deepcopy(masked_list)
        transitive_matrix = torch.diag(torch.zeros(d)).reshape(-1)
        transitive_matrix[transitive_list[0]] = 1
        transitive_matrix = [transitive_matrix.reshape(d, d)] * mbsize
        updated_order = [torch.Tensor([])] * mbsize
        while not all(done):
            with torch.no_grad():
                model_outcome = self.pol(s)

                for i in range(len(s)):
                    item = model_outcome[i]
                    item[masked_list_new[i]] = 0   # mask_matrix

                pol = Categorical(probs=model_outcome)

                acts = pol.sample()

            step_full = [self.envs.step_new(a, state, tm, d, order) for a, state, tm, d, order in
                         zip(acts, s, transitive_matrix, done, updated_order)]

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            for si,a,(sp, r, d, m_list, ini_done, updated_mt, true_step, updated_order),(traj_idx, _) in zip(s, acts, step_full, sorted(m.items())):
                trajs[traj_idx].append( [si[None, :]] + [i for i in (a.unsqueeze(0), tf([r]), sp.unsqueeze(0), tf([d]))] )

            done = [bool(d or step_full[m[i]][2]) for i, d in enumerate(done)]

            updated_order = []
            states = []
            masked_list_new = []
            transitive_matrix = []
            for item in step_full:

                updated_order.append(item[7][0])

                states.append(np.array(item[0]))

                masked_list_new.append(item[3])

                transitive_matrix.append(item[5])

            states = tf(np.array(states))
            states = states.to(args.dev)

        return sum(trajs.values(), []), states, updated_order

    def learn_from(self,batch):
        s, a, r, sp, d = [torch.cat(i,0) for i in zip(*batch)]
        ar = torch.arange(s.shape[0])
        a = a.long()
        d = d.unsqueeze(1)
        q1 = self.Q_1(s)
        q1a = q1[ar,a]
        q2 = self.Q_2(s)
        q2a = q2[ar,a]

        ps = self.pol(s)
        with torch.no_grad():
            qt1 = self.Q_t1(sp)
            qt2 = self.Q_t2(sp)
            psp = self.pol(sp)
        vsp1 = ((1 - d) * psp * (qt1 - self.alpha * torch.log(psp))).sum(1)
        vsp2 = ((1 - d) * psp * (qt2 - self.alpha * torch.log(psp))).sum(1)
        J_Q = (0.5 * (q1a - r - vsp1).pow(2) + 0.5 * (q2a - r - vsp2).pow(2)).mean()
        minq = torch.min(q1, q2).detach()
        J_pi = (ps * (self.alpha * torch.log(ps) - minq)).sum(1).mean()
        J_alpha = (ps.detach() * (-self.alpha * torch.log(ps.detach()) + self.alpha_target)).sum(1).mean()

        for A, B in [(self.Q_1, self.Q_t1), (self.Q_2, self.Q_t2)]:
            for a, b in zip(A.parameters(), B.parameters()):
                b.data.mul_(1 - self.tau).add_(self.tau * a)
        return J_Q + J_pi + J_alpha, J_Q, J_pi, J_alpha, self.alpha
