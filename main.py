
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


class FlowNetAgent_cycleness:
    def __init__(self, args, envs, reward_):
        self.n_node = args.n_node
        self.xdim = args.n_node * args.n_node
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.reward_ = reward_


        self.model = make_mlp([self.xdim] +
                              [args.n_hid] * args.n_layers + [self.xdim])

        self.model.to(args.dev)
        self.target = copy.deepcopy(self.model)
        self.envs = envs
        self.tau = args.bootstrap_tau
        self.replay = ReplayBuffer(args, envs)
        self.em = torch.nn.Embedding(self.xdim, self.embed_dim, padding_idx=None, max_norm=None, norm_type=2.0)

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, s, masked_list):
        batch = []
        d = args.n_node
        batch += self.replay.sample()
        masked_list_new = copy.deepcopy(masked_list)
        done = [False] * mbsize

        updated_order = [torch.Tensor([])] * mbsize
        while not all(done):
            with torch.no_grad():
                model_outcome = self.model(s)

            for i in range(len(s)):
                item = model_outcome[i]
                item[masked_list_new[i]] -= (1e20)

            acts = Categorical(logits=model_outcome).sample()

            step_full = [self.envs.step_cycleness(a, state, mask_list) for a, state,mask_list in
                         zip(acts, s,masked_list_new)]


            p_a = [
                self.envs.parent_transitions(sp)
                for a, (sp, r, done, m_list, ini_done, transitive_m, true_step, updated_order) in
                zip(acts, step_full) if not ini_done]

            for (p, a), (sp, r, d, m_list, ini_done, updated_mt, true_step, updated_order) in zip(p_a, step_full):
                if not ini_done:
                    if d:
                        sp = true_step
                    batch += [[p[0].unsqueeze(0), tf(np.array([a[0]])), tf([r]), sp.unsqueeze(0), tf(np.array([d]))]]

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step_full[m[i]][2]) for i, d in enumerate(done)]

            updated_order = []
            states = []
            masked_list_new = []

            for item in step_full:

                states.append(np.array(item[0]))

                masked_list_new.append(item[3])

            states = tf(np.array(states))
            states = states.to(args.dev)
            s = states

            if args.replay_strategy == "top_k":
                for (sp, r, d, m_list, ini_done, updated_mt, true_step, order) in step_full:
                    self.replay.add(tuple(sp), r)

        return batch, states, None

    def learn_from(self, batch):

        loginf = torch.Tensor([1000])
        batch_idxs = tl(sum([[i] * len(parents) for i, (parents, _, _, _, _) in enumerate(batch)], []))

        parents, actions, r, sp, done = map(torch.cat, zip(*batch))
        parents = parents.to(args.dev)
        actions = actions.to(args.dev)

        if args.model_name == 'CNN':
            batch_parents = len(parents)
            cnn_parents = parents.reshape(batch_parents, 1, self.n_node, self.n_node).to(torch.float64)

            batch_sp = len(sp)
            sp = sp.reshape(batch_sp, 1, self.n_node, self.n_node)
            parents_Qsa = self.model(cnn_parents.clone().detach().requires_grad_(True))[
                torch.arange(parents.shape[0]), actions.long()]
            cnn_sp = sp.reshape(batch_sp, 1, self.n_node, self.n_node).to(torch.float64)
            next_q = self.model(cnn_sp.clone().detach().requires_grad_(True))
            parents_Qsa = parents_Qsa.cpu()

            in_flow = torch.log(torch.zeros(sp.shape[0]).index_add_(0, batch_idxs, torch.exp(parents_Qsa)))
        if args.model_name == 'MLP':
            parents_Qsa = self.model(parents.clone().detach().requires_grad_(True))[
                torch.arange(parents.shape[0]), actions.long()]
            next_q = self.model(sp.clone().detach().requires_grad_(True))
            parents_Qsa = parents_Qsa.cpu()

            in_flow = torch.log(torch.zeros(sp.shape[0]).index_add_(0, batch_idxs, torch.exp(parents_Qsa)))

        elif args.model_name == 'MLP_encode':
            parents = parents.int().to(args.dev)
            parents_em_s = self.em(parents).permute(1, 0, 2)
            parents_output = self.model(parents_em_s).permute(1, 0, 2)

            model_outcome = torch.mean(parents_output, axis=-1)

            parents_Qsa = model_outcome[torch.arange(parents.shape[0]), actions.long()]

            sp = sp.int().to(args.dev)
            sp = self.em(sp)
            next_q = self.model(sp.clone().detach().requires_grad_(True))
            next_q = torch.mean(next_q, axis=-1)

            parents_Qsa = parents_Qsa.cpu()

            in_flow = torch.log(torch.zeros(sp.shape[0]).index_add_(0, batch_idxs, torch.exp(parents_Qsa)))

        next_q = next_q.cpu()

        next_qd = (1 - done).unsqueeze(1) * next_q + done.unsqueeze(1) * (-loginf)

        out_flow = torch.logsumexp(torch.cat([torch.log(r)[:, None], next_qd], 1), 1)

        loss = (in_flow - out_flow).pow(2).mean()

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (done.sum() + 1e-20)
            flow_loss = ((in_flow - out_flow) * (1 - done)).pow(2).sum() / ((1 - done).sum() + 1e-20)

        return loss, term_loss, flow_loss

    
    
def step_cycleness(self,a,s,mask_list):
    done = False
    d = args.n_node

    new_s = copy.deepcopy(s)
    new_s[int(a)] = 1
    matrix_new_s = new_s.reshape(d, d)  # update the state
    action_row = int(a) // args.n_node
    action_column = int(a) % args.n_node

    # update_masked_matrix

    updated_mask = copy.deepcopy(matrix_new_s)
    updated_mask[action_column,action_row] = 1
    new_mask_position = torch.where(updated_mask.reshape(-1)==1)[0].numpy()
    mask_list = torch.cat( (mask_list , torch.Tensor(new_mask_position)),0).long()
    max_edges = int(  args.n_node * (args.n_node-1)/2)
    number_edges = torch.sum(new_s).item()

    if number_edges == max_edges:
        done = True
        new_r = self.reward_.cal_BIC_cycleness(matrix_new_s, X) # We add some penalties in this BIC score
        new_s = matrix_new_s.reshape(-1)
        true_step_matrix = copy.deepcopy(matrix_new_s)
        true_s = true_step_matrix.reshape(-1)
    return new_s, new_r if done else 0, done, mask_list, None, None, true_s if done else None, None
