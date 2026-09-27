/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

/* Conforming update rows at a refinement interface, after the prototype's
   Box3D.  One grid of edges (E) and faces (H): coarse samples outside and on
   the fine region's boundary, fine ones strictly inside it; a fine E sample on
   the boundary is slaved -- replicated along its own edge, linear across it
   where it falls between coarse ones.  H = C E and E = -W_E^-1 C^T W_H H with W
   the exactly rasterised dual volumes, so the pair is adjoint by construction.
   In 2-D this is TM (Ez a node, linearly slaved) and TE (Ex, Ey edges) at once.

   At the interface these rows replace only the curl in meep's step of D and B;
   the PML and conductivity recursions around it, and everything that derives E
   and H, are meep's own.  Elsewhere meep's update is already the conforming one.

   Two resolutions.  Chunks of different resolution have no ghost connections
   between them; the rows set or derive every ghost that is read. */

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "meep.hpp"
#include "meep_internals.hpp"

using namespace std;

namespace meep {

struct conforming_rows {
  struct loc {
    int chunk;
    component c;
    ptrdiff_t idx;
    ivec here;
    bool owned;
    char rank;                     // 0 owned, 1 a ghost inside its chunk, 2 a ghost beyond it
    std::complex<double> ph = 1.0; // Bloch phase of this copy over the dof
  };
  struct term {
    int k;
    double w;
    std::complex<double> ph = 1.0; // Bloch phase of the dof where the term reads it
  };
  bool bloch = false; // some phase is not 1: values are complex throughout
  // what the step carries at a point: D or B, its PML and conductivity
  // auxiliaries, and E or H with its PML auxiliary W
  struct state {
    double f, u, cnd, h, w;
  };

  int ncmp = 1;
  int components = 0;            // which field components were allocated, on any process
  int local_components = 0;      // ... on this one
  vector<fields_chunk *> layout; // the chunks these rows were built for
  vector<vector<loc> > where;    // every stored copy of each dof, owned first
  vector<int> hdof, edof;        // dofs whose curl is replaced
  vector<vector<term> > hrow, erow;
  vector<vector<loc> > swhere; // slaved positions
  vector<vector<term> > srow;
  vector<char> smagnetic;
  vector<state> hold[2], eold[2]; // at the start of the step
  // where the rows derive E or H: each polarization's P and P_prev there
  vector<vector<realnum> > hpol[2], epol[2];
  vector<state> hnew[2], enew[2]; // D or B after it
  vector<char> hderive, ederive;  // no chunk owns the dof, so meep never derives it
  /* Across processes: each dof has one site, its best stored copy anywhere;
     only the site's process steps it, and every dof's value is exchanged at
     the three points of a step where the rows read or spread. */
  vector<char> site;     // this process holds the dof's site
  vector<double> val[2]; // every dof's E or H, then its D or B, on every process

  static component stepped(component c) {
    return direction_component(is_electric(c) ? Dx : Bx, component_direction(c));
  }
  double read(fields &, int k, int cmp) const { return val[cmp][k]; }
  std::complex<double> value(int k, bool st) const {
    const size_t j = st ? where.size() + k : k;
    return std::complex<double>(val[0][j], ncmp > 1 ? val[1][j] : 0.0);
  }
  // a row's sum, as the dof's site holds it
  double row_sum(const vector<term> &row, int k, int cmp) const {
    if (!bloch) {
      double acc = 0;
      for (const term &q : row)
        acc += q.w * val[cmp][q.k];
      return acc;
    }
    std::complex<double> acc = 0;
    for (const term &q : row)
      acc += q.w * q.ph * value(q.k, false);
    acc *= where[k][0].ph;
    return cmp ? imag(acc) : real(acc);
  }
  double read_stepped(int k, int cmp) const { return val[cmp][where.size() + k]; }
  // each site's E or H, then D or B, as pointers; retaken when any array moves
  vector<const realnum *> arrays, at[2];
  // per peer process, in dof order: the sites sent to it, the dofs received from it
  vector<int> owner, peers;
  vector<vector<int> > sends, recvs;
  vector<realnum> outbuf, inbuf;

  /* Sources on the interface are the rows': meep's amplitude is per cell of
     the chunk holding it, the rows' per dual volume of a dof, and a slave's
     goes to its masters (the adjoint of the slave rule). */
  struct source {
    int k;
    const src_time *t;
    std::complex<double> a;
    std::complex<double> off = 0; // a dipole taken off D for this step's update_eh
  };
  struct taken { // an amplitude set to zero in meep's list, restored when the rows go
    fields_chunk *chunk;
    int ft;
    size_t sv, j;
    const src_time *t;
    std::complex<double> a;
  };
  int nd = 2;
  vector<source> sources; // at this process's sites
  vector<taken> zeroed;
  vector<double> vol;                       // each dof's dual volume
  vector<int> hrow_of, erow_of;             // each dof's row, or -1
  map<array<int, 4>, vector<term> > src_to; // a stored position -> the dofs it feeds
  std::function<array<int, 4>(component, const ivec &)> key_of;
  std::function<std::complex<double>(const ivec &)> phase_of;
  double src_sig = -1;

  static double source_signature(fields &f) {
    double s = 0;
    for (int i = 0; i < f.num_chunks; ++i)
      if (f.chunks[i]->is_mine())
        for (int ft : {B_stuff, D_stuff})
          for (const src_vol &sv : f.chunks[i]->sources[ft]) {
            s += 1 + 1e-3 * sv.num_points();
            for (size_t j = 0; j < sv.num_points(); ++j)
              s += std::abs(sv.amplitude_at(j));
          }
    return s;
  }
  void take_sources(fields &f) {
    vector<const src_time *> list;
    for (const src_time *s = f.sources; s; s = s->next)
      list.push_back(s);
    auto gone = [&](const src_time *s) {
      return std::find(list.begin(), list.end(), s) == list.end();
    };
    sources.erase(
        std::remove_if(sources.begin(), sources.end(), [&](const source &s) { return gone(s.t); }),
        sources.end());
    zeroed.erase(
        std::remove_if(zeroed.begin(), zeroed.end(), [&](const taken &z) { return gone(z.t); }),
        zeroed.end());
    vector<double> mine; // dof, source number, re, im
    for (int i = 0; i < f.num_chunks; ++i) {
      if (!f.chunks[i]->is_mine()) continue;
      fields_chunk *fc = f.chunks[i];
      const grid_volume &gv = fc->gv;
      for (int ft : {B_stuff, D_stuff})
        for (size_t v = 0; v < fc->sources[ft].size(); ++v) {
          src_vol &sv = fc->sources[ft][v];
          const direction d = component_direction(sv.c);
          const component ec = is_D(sv.c)   ? direction_component(Ex, d)
                               : is_B(sv.c) ? direction_component(Hx, d)
                                            : sv.c;
          const double tn = double(std::find(list.begin(), list.end(), sv.t()) - list.begin());
          for (size_t j = 0; j < sv.num_points(); ++j) {
            const std::complex<double> a = sv.amplitude_at(j);
            if (a == 0.0) continue;
            const ivec here = gv.iloc(sv.c, sv.index_at(j));
            auto it = src_to.find(key_of(ec, here));
            if (it == src_to.end()) continue;
            // meep's amplitude is per cell of this chunk, and in this copy's frame
            std::complex<double> dipole = a * conj(phase_of(here));
            for (int n = 0; n < nd; ++n)
              dipole /= gv.a;
            for (const term &q : it->second) {
              const std::complex<double> b = dipole * q.w * conj(q.ph) / vol[q.k];
              mine.insert(mine.end(), {double(q.k), tn, real(b), imag(b)});
            }
            zeroed.push_back(taken{fc, ft, v, j, sv.t(), a});
            sv.set_amplitude(j, 0.0);
          }
        }
    }
    vector<double> all(mine);
    if (count_processors() > 1) { // every contribution to every process; each keeps its sites'
      const int n = (int)mine.size(), total = sum_to_all(n), end = partial_sum_to_all(n);
      vector<double> buf(total, 0.0);
      std::copy(mine.begin(), mine.end(), buf.begin() + (end - n));
      all.assign(total, 0.0);
      sum_to_all(buf.data(), all.data(), total);
    }
    for (size_t e = 0; e + 3 < all.size(); e += 4) {
      const int k = (int)all[e];
      if (!site[k]) continue;
      const src_time *s = list[(size_t)all[e + 1]];
      const std::complex<double> a(all[e + 2], all[e + 3]);
      auto it = std::find_if(sources.begin(), sources.end(),
                             [&](const source &x) { return x.k == k && x.t == s; });
      if (it != sources.end())
        it->a += a;
      else
        sources.push_back(source{k, s, a, 0.0});
    }
    src_sig = source_signature(f);
  }
  void restore_sources(fields &f) {
    for (const taken &z : zeroed) {
      if (std::find(f.chunks, f.chunks + f.num_chunks, z.chunk) == f.chunks + f.num_chunks)
        continue;
      vector<src_vol> &svs = z.chunk->sources[z.ft];
      if (z.sv < svs.size() && svs[z.sv].t() == z.t && z.j < svs[z.sv].num_points())
        svs[z.sv].set_amplitude(z.j, svs[z.sv].amplitude_at(z.j) + z.a);
    }
    zeroed.clear();
  }
  // dft_ldos's sums: each source's J per cell of f.gv, as meep's own are
  void ldos_terms(const fields &f, std::complex<double> &EJ, std::complex<double> &HJ,
                  double &Jsum) const {
    for (const source &s : sources) {
      const loc &l = where[s.k][0];
      const fields_chunk *fc = f.chunks[l.chunk];
      std::complex<double> A = s.a * vol[s.k];
      for (int n = 0; n < nd; ++n)
        A *= f.gv.a;
      A *= l.ph;
      const realnum *re = fc->f[l.c][0], *im = fc->f[l.c][1];
      if (!re) continue;
      const std::complex<double> v(re[l.idx], im ? im[l.idx] : 0.0);
      (is_magnetic(l.c) ? HJ : EJ) += v * conj(A);
      Jsum += std::abs(A);
    }
  }
  /* Right where meep adds sources: D or B -= dt J.  A source given as a
     dipole p is meep's E = eps^-1 (D - p): p comes off D for update_eh only,
     and goes back on right after it (unshift), or, where the rows derive E
     themselves, after they have (restore_derived). */
  bool derived(int k, bool magnetic) const {
    const int r = (magnetic ? hrow_of : erow_of)[k];
    return r >= 0 && (magnetic ? hderive : ederive)[r];
  }
  void shift(fields &f, const source &s, std::complex<double> d, bool array, bool state) {
    const loc &l = where[s.k][0];
    d *= l.ph;
    fields_chunk *fc = f.chunks[l.chunk];
    const component cc = stepped(l.c);
    const bool magnetic = is_magnetic(l.c);
    const int r = (magnetic ? hrow_of : erow_of)[s.k];
    for (int cmp = 0; cmp < ncmp; ++cmp) {
      const double v = cmp ? imag(d) : real(d);
      realnum *a = fc->f[cc][cmp] ? fc->f[cc][cmp] : fc->f[l.c][cmp];
      if (array) a[l.idx] += (realnum)v;
      if (state && r >= 0) (magnetic ? hnew : enew)[cmp][r].f += v;
    }
  }
  void apply_sources(fields &f, bool magnetic) {
    for (source &s : sources) {
      const loc &l = where[s.k][0];
      if (is_magnetic(l.c) != magnetic) continue;
      if (s.t->is_integrated) {
        s.off = s.a * s.t->dipole(f.time() + (magnetic ? 0.5 : 1.0) * f.dt);
        shift(f, s, -s.off, true, derived(s.k, magnetic));
        continue;
      }
      std::complex<double> dd = -s.a * f.dt * s.t->current();
      const component cc = stepped(l.c);
      if (const realnum *ci = f.chunks[l.chunk]->s->condinv[cc][component_direction(cc)])
        dd *= ci[l.idx];
      shift(f, s, dd, true, true);
    }
  }
  void unshift(fields &f, bool magnetic) {
    for (const source &s : sources)
      if (s.t->is_integrated && is_magnetic(where[s.k][0].c) == magnetic && !derived(s.k, magnetic))
        shift(f, s, s.off, true, false);
  }
  void restore_derived(fields &f, bool magnetic) {
    for (const source &s : sources)
      if (s.t->is_integrated && is_magnetic(where[s.k][0].c) == magnetic && derived(s.k, magnetic))
        shift(f, s, s.off, true, true);
  }
  void point_at_sites(fields &f) {
    vector<const realnum *> now;
    for (int i = 0; i < f.num_chunks; ++i)
      FOR_COMPONENTS(c) {
        for (int cmp = 0; cmp < 2; ++cmp)
          now.push_back(f.chunks[i]->f[c][cmp]);
      }
    if (now == arrays) return;
    arrays.swap(now);
    const size_t nd = where.size();
    for (int cmp = 0; cmp < ncmp; ++cmp) {
      at[cmp].assign(2 * nd, NULL);
      for (size_t k = 0; k < nd; ++k) {
        if (!site[k]) continue;
        const loc &l = where[k][0];
        realnum *const *a = f.chunks[l.chunk]->f[l.c], *const *s =
                                                           f.chunks[l.chunk]->f[stepped(l.c)];
        if (a[cmp]) at[cmp][k] = a[cmp] + l.idx;
        if (s[cmp] || a[cmp]) at[cmp][nd + k] = (s[cmp] ? s[cmp] : a[cmp]) + l.idx;
      }
    }
  }
  void exchange(fields &f) {
    point_at_sites(f);
    const size_t nd = where.size(), w = 2 * ncmp; // values per dof
    for (int cmp = 0; cmp < ncmp; ++cmp) {
      val[cmp].resize(2 * nd);
      for (size_t k = 0; k < 2 * nd; ++k)
        if (at[cmp][k]) val[cmp][k] = *at[cmp][k];
    }
    if (bloch) // a site holds the dof's value times its phase
      for (size_t k = 0; k < nd; ++k)
        if (site[k] && where[k][0].ph != 1.0)
          for (size_t j : {k, nd + k}) {
            const std::complex<double> v =
                std::complex<double>(val[0][j], val[1][j]) * conj(where[k][0].ph);
            val[0][j] = real(v);
            val[1][j] = imag(v);
          }
    if (peers.empty()) return;
    size_t nout = 0, nin = 0;
    for (size_t i = 0; i < peers.size(); ++i) {
      nout += w * sends[i].size();
      nin += w * recvs[i].size();
    }
    outbuf.resize(nout);
    inbuf.resize(nin);
    {
      std::unique_ptr<comms_manager> m = create_comms_manager();
      realnum *in = inbuf.data(), *out = outbuf.data();
      for (size_t i = 0; i < peers.size(); in += w * recvs[i++].size())
        if (!recvs[i].empty()) m->receive_real_async(in, w * recvs[i].size(), peers[i], 0, [] {});
      for (size_t i = 0; i < peers.size(); ++i) {
        realnum *start = out;
        for (int k : sends[i])
          for (int cmp = 0; cmp < ncmp; ++cmp) {
            *out++ = (realnum)val[cmp][k];
            *out++ = (realnum)val[cmp][nd + k];
          }
        if (out > start) m->send_real_async(start, out - start, peers[i], 0);
      }
    } // waits for every transfer
    const realnum *in = inbuf.data();
    for (size_t i = 0; i < peers.size(); ++i)
      for (int k : recvs[i])
        for (int cmp = 0; cmp < ncmp; ++cmp) {
          val[cmp][k] = *in++;
          val[cmp][nd + k] = *in++;
        }
  }
  state save(fields &f, const loc &l, int cmp) const {
    const fields_chunk *fc = f.chunks[l.chunk];
    const component cc = stepped(l.c);
    const realnum *F = fc->f[cc][cmp] ? fc->f[cc][cmp] : fc->f[l.c][cmp];
    const realnum *U = fc->f_u[cc][cmp], *K = fc->f_cond[cc][cmp];
    const realnum *E = fc->f[l.c][cmp], *W = fc->f_w[l.c][cmp];
    const double v = F ? F[l.idx] : 0.0, e = E ? E[l.idx] : v;
    // meep allocates U and W as copies of F and E, and K as zero
    return state{v, U ? U[l.idx] : v, K ? K[l.idx] : 0.0, e, W ? W[l.idx] : e};
  }
  /* The owned copy's D or B after a step whose curl term is c = dt curl,
     exactly as step_curl computes it. */
  state step_point(fields &f, const loc &l, int cmp, const state &o, double c) const {
    fields_chunk *fc = f.chunks[l.chunk];
    const structure_chunk *s = fc->s;
    const grid_volume &gv = fc->gv;
    const component cc = stepped(l.c);
    const direction dc = component_direction(cc);
    const direction ds = cycle_direction(gv.dim, dc, 1), du = cycle_direction(gv.dim, dc, 2);
    const bool has_s = s->sigsize[ds] > 1, has_u = s->sigsize[du] > 1;
    const int k = l.here.in_direction(ds) - gv.little_corner().in_direction(ds);
    const int ku = l.here.in_direction(du) - gv.little_corner().in_direction(du);
    if ((has_s && (k < 0 || k >= s->sigsize[ds])) || (has_u && (ku < 0 || ku >= s->sigsize[du])))
      meep::abort("conforming rows: PML index outside its chunk's array");
    const realnum *cnd = s->conductivity[cc][dc], *cndinv = s->condinv[cc][dc];
    const double dt2 = 0.5 * f.dt;
    double F = o.f, U = o.u, K = o.cnd;
    // the curl term after conductivity, entering f, u, or the cond auxiliary
    auto driven = [&](double prev) {
      return cnd ? ((1 - dt2 * cnd[l.idx]) * prev + c) * cndinv[l.idx] : prev + c;
    };
    double into; // what enters the sigma-in-f stage, or f itself
    if (has_s) {
      double inc = c;
      if (cnd) {
        const double K0 = K;
        K = driven(K0);
        inc = K - K0;
      }
      const double g = has_u ? U : F;
      into = ((s->kap[ds][k] - s->sig[ds][k]) * g + inc) * s->siginv[ds][k];
    }
    else
      into = driven(has_u ? U : F);
    if (has_u) {
      F = s->siginv[du][ku] * ((s->kap[du][ku] - s->sig[du][ku]) * F + into - U);
      U = into;
    }
    else
      F = into;
    realnum *A = fc->f[cc][cmp] ? fc->f[cc][cmp] : fc->f[l.c][cmp];
    A[l.idx] = (realnum)F;
    if (has_u && fc->f_u[cc][cmp]) fc->f_u[cc][cmp][l.idx] = (realnum)U;
    if (has_s && cnd && fc->f_cond[cc][cmp]) fc->f_cond[cc][cmp][l.idx] = (realnum)K;
    return state{F, U, K, o.h, o.w};
  }
  /* E or H from D or B where no chunk does it, as update_eh would: diagonal,
     linear, with the PML auxiliary W.  The exchange has overwritten this
     ghost since, so its state is restored first. */
  void derive(fields &f, const loc &l, int cmp, const state &o, const state &n,
              vector<realnum> &P) const {
    fields_chunk *fc = f.chunks[l.chunk];
    const structure_chunk *s = fc->s;
    const grid_volume &gv = fc->gv;
    const component cc = stepped(l.c);
    realnum *F = fc->f[cc][cmp], *E = fc->f[l.c][cmp];
    if (fc->f_u[cc][cmp]) fc->f_u[cc][cmp][l.idx] = (realnum)n.u;
    if (fc->f_cond[cc][cmp]) fc->f_cond[cc][cmp][l.idx] = (realnum)n.cnd;
    if (F) F[l.idx] = (realnum)n.f;
    const direction d = component_direction(l.c);
    const realnum *chi = s->chi1inv[l.c][d];
    double dmp = n.f; // D - P, as update_eh subtracts it
    for (size_t k = 0; k < P.size(); k += 2)
      dmp -= P[k];
    const double g = dmp * (chi ? chi[l.idx] : 1.0);
    double e = g;
    if (s->sigsize[d] > 1) {
      const int kw = l.here.in_direction(d) - gv.little_corner().in_direction(d);
      if (kw < 0 || kw >= s->sigsize[d])
        meep::abort("conforming rows: PML index outside its chunk's array");
      e = o.h + (s->kap[d][kw] + s->sig[d][kw]) * g - (s->kap[d][kw] - s->sig[d][kw]) * o.w;
      if (realnum *W = fc->f_w[l.c][cmp]) W[l.idx] = (realnum)g;
    }
    if (E) E[l.idx] = (realnum)e;
    // then update_pols: each P from W (E outside PML); synchronize_magnetic_fields has none
    if (f.subgrid_syncing) return;
    const double w = s->sigsize[d] > 1 ? g : e;
    size_t k = 0;
    for (const polarization_state *p = fc->pol[type(l.c)]; p; p = p->next, k += 2) {
      const realnum *sg = p->s->sigma[l.c][d];
      if (!sg || sg[l.idx] == 0) continue; // P stays 0
      if (!p->s->update_P_point((realnum)w, sg[l.idx], (realnum)f.dt, P[k], P[k + 1]))
        meep::abort("conforming rows: a susceptibility the rows cannot step at one point");
    }
  }
  // after meep has derived E or H at the site: every copy the same, D and B too
  void spread(fields &f, int k, int cmp) const {
    if (bloch) {
      if (cmp) return; // both parts at once
      const std::complex<double> v = value(k, false), vs = value(k, true);
      for (size_t i = site[k] ? 1 : 0; i < where[k].size(); ++i) {
        const loc &l = where[k][i];
        const std::complex<double> u = l.ph * v, us = l.ph * vs;
        realnum **a = f.chunks[l.chunk]->f[l.c], **b = f.chunks[l.chunk]->f[stepped(l.c)];
        for (int m = 0; m < 2; ++m) {
          if (a[m]) a[m][l.idx] = (realnum)(m ? imag(u) : real(u));
          if (b[m] && b[m] != a[m]) b[m][l.idx] = (realnum)(m ? imag(us) : real(us));
        }
      }
      return;
    }
    const double v = read(f, k, cmp), vs = read_stepped(k, cmp);
    for (size_t i = site[k] ? 1 : 0; i < where[k].size(); ++i) {
      const loc &l = where[k][i];
      realnum *a = f.chunks[l.chunk]->f[l.c][cmp], *b = f.chunks[l.chunk]->f[stepped(l.c)][cmp];
      if (a) a[l.idx] = (realnum)v;
      if (b && b != a) b[l.idx] = (realnum)vs;
    }
  }
  void slaves(fields &f, bool magnetic) {
    if (bloch) {
      for (size_t s = 0; s < srow.size(); ++s) {
        if (smagnetic[s] != magnetic) continue;
        std::complex<double> v = 0, vs = 0;
        for (const term &q : srow[s]) {
          v += q.w * q.ph * value(q.k, false);
          vs += q.w * q.ph * value(q.k, true);
        }
        for (const loc &l : swhere[s]) {
          realnum **a = f.chunks[l.chunk]->f[l.c], **b = f.chunks[l.chunk]->f[stepped(l.c)];
          const std::complex<double> u = l.ph * v, us = l.ph * vs;
          for (int cmp = 0; cmp < 2; ++cmp) {
            if (a[cmp]) a[cmp][l.idx] = (realnum)(cmp ? imag(u) : real(u));
            if (b[cmp] && b[cmp] != a[cmp]) b[cmp][l.idx] = (realnum)(cmp ? imag(us) : real(us));
          }
        }
      }
      return;
    }
    for (int cmp = 0; cmp < ncmp; ++cmp)
      for (size_t s = 0; s < srow.size(); ++s) {
        if (smagnetic[s] != magnetic) continue;
        double v = 0, vs = 0;
        for (const term &q : srow[s]) {
          v += q.w * read(f, q.k, cmp);
          vs += q.w * read_stepped(q.k, cmp);
        }
        for (const loc &l : swhere[s]) {
          realnum **a = f.chunks[l.chunk]->f[l.c], **b = f.chunks[l.chunk]->f[stepped(l.c)];
          if (a[cmp]) a[cmp][l.idx] = (realnum)v;
          if (b[cmp] && b[cmp] != a[cmp]) b[cmp][l.idx] = (realnum)vs; // for E.D and H.B
        }
      }
  }
  /* The site of each dof: its best-ranked stored copy on any process, ties to
     the lowest process. */
  void assign_sites() {
    const size_t nd = where.size();
    const int np = count_processors(), me = my_rank();
    if (np > 64) meep::abort("conforming rows: at most 64 processes so far");
    vector<size_t> level(4 * nd, 0), all_level(4 * nd, 0);
    for (size_t k = 0; k < nd; ++k)
      level[4 * k + (where[k].empty() ? 3 : where[k][0].rank)] = 1;
    sum_to_all(level.data(), all_level.data(), (int)level.size());
    vector<size_t> bits(nd, 0), all_bits(nd, 0);
    for (size_t k = 0; k < nd; ++k) {
      int best = 0;
      while (best < 3 && !all_level[4 * k + best])
        ++best;
      if (best == 3) meep::abort("conforming rows: a dof with no storage anywhere");
      if (!where[k].empty() && where[k][0].rank == best) bits[k] = size_t(1) << me;
    }
    bw_or_to_all(bits.data(), all_bits.data(), (int)nd);
    site.assign(nd, 0);
    owner.assign(nd, 0);
    for (size_t k = 0; k < nd; ++k) {
      while (!(all_bits[k] >> owner[k] & 1)) // the lowest process holding it
        ++owner[k];
      site[k] = owner[k] == me;
    }
  }
  /* Who sends what to whom: a process needs a dof's value where it stores a
     copy, or where its rows or slaves read it. */
  void plan_exchange() {
    const size_t nd = where.size();
    const int np = count_processors(), me = my_rank();
    peers.clear();
    sends.clear();
    recvs.clear();
    if (np == 1) return;
    vector<char> need(nd, 0);
    for (size_t k = 0; k < nd; ++k)
      need[k] = !where[k].empty();
    for (int e = 0; e < 2; ++e) {
      const vector<int> &dofs = e ? edof : hdof;
      const vector<vector<term> > &rows = e ? erow : hrow;
      for (size_t i = 0; i < dofs.size(); ++i)
        if (site[dofs[i]])
          for (const term &q : rows[i])
            need[q.k] = 1;
    }
    for (size_t s = 0; s < srow.size(); ++s)
      if (!swhere[s].empty())
        for (const term &q : srow[s])
          need[q.k] = 1;
    vector<size_t> bits(nd, 0), all_bits(nd, 0);
    for (size_t k = 0; k < nd; ++k)
      if (need[k]) bits[k] = size_t(1) << me;
    bw_or_to_all(bits.data(), all_bits.data(), (int)nd);
    for (int q = 0; q < np; ++q) {
      if (q == me) continue;
      vector<int> s, r;
      for (size_t k = 0; k < nd; ++k) {
        if (owner[k] == me && (all_bits[k] >> q & 1)) s.push_back(k);
        if (owner[k] == q && need[k]) r.push_back(k);
      }
      if (s.empty() && r.empty()) continue;
      peers.push_back(q);
      sends.push_back(s);
      recvs.push_back(r);
    }
  }
  void finish(fields &f) {
    assign_sites();
    plan_exchange();
    hrow_of.assign(where.size(), -1);
    erow_of.assign(where.size(), -1);
    for (size_t i = 0; i < hdof.size(); ++i)
      hrow_of[hdof[i]] = (int)i;
    for (size_t i = 0; i < edof.size(); ++i)
      erow_of[edof[i]] = (int)i;
    for (int e = 0; e < 2; ++e) {
      const vector<int> &dofs = e ? edof : hdof;
      vector<char> &derived = e ? ederive : hderive;
      derived.assign(dofs.size(), 0);
      for (size_t i = 0; i < dofs.size(); ++i) {
        if (!site[dofs[i]]) continue;
        const loc &l = where[dofs[i]][0];
        if (l.owned) continue;
        derived[i] = 1;
        const fields_chunk *fc = f.chunks[l.chunk];
        if (fc->s->chi2[l.c] || fc->s->chi3[l.c])
          meep::abort("conforming rows: no nonlinearity where no chunk owns an interface dof "
                      "yet");
        const direction d = component_direction(l.c);
        for (const polarization_state *p = fc->pol[type(l.c)]; p; p = p->next) {
          const realnum *sg = p->s->sigma[l.c][d];
          realnum q = 0, qq = 0;
          if (sg && sg[l.idx] != 0 && !p->s->update_P_point(0, 0, 1, q, qq))
            meep::abort("conforming rows: a susceptibility the rows cannot step where no chunk "
                        "owns an interface dof (Lorentzian and Drude only)");
          FOR_DIRECTIONS(d2) {
            if (d2 != d && p->s->sigma[l.c][d2] && p->s->sigma[l.c][d2][l.idx] != 0)
              meep::abort("conforming rows: no anisotropic dispersion where no chunk owns an "
                          "interface dof yet");
          }
        }
      }
    }
    for (int cmp = 0; cmp < 2; ++cmp)
      for (int e = 0; e < 2; ++e) {
        const vector<int> &dofs = e ? edof : hdof;
        vector<vector<realnum> > &P = e ? epol[cmp] : hpol[cmp];
        P.assign(dofs.size(), vector<realnum>());
        for (size_t i = 0; i < dofs.size(); ++i) {
          if (!(e ? ederive : hderive)[i]) continue;
          const loc &l = where[dofs[i]][0];
          for (const polarization_state *p = f.chunks[l.chunk]->pol[type(l.c)]; p; p = p->next)
            P[i].insert(P[i].end(), {0, 0});
        }
      }
    for (int cmp = 0; cmp < 2; ++cmp) {
      hold[cmp].resize(hdof.size());
      eold[cmp].resize(edof.size());
      hnew[cmp].resize(hdof.size());
      enew[cmp].resize(edof.size());
    }
    exchange(f);
    slaves(f, true);
    slaves(f, false);
  }
};

namespace {

typedef conforming_rows::loc sg_loc;
typedef conforming_rows::term sg_term;
typedef array<int, 3> sg_pt;  // a lattice position; z = 0 in 2-D
typedef array<int, 4> sg_key; // component, then its lattice position

const component all_components[6] = {Ex, Ey, Ez, Hx, Hy, Hz};

int component_signature(const fields &f) {
  int s = 0;
  for (int i = 0; i < f.num_chunks; ++i)
    FOR_COMPONENTS(c) {
      if (f.chunks[i]->f[c][0]) s |= 1 << c;
    }
  return s;
}

int global_signature(int local) {
  size_t in = (size_t)local, out = 0;
  bw_or_to_all(&in, &out, 1);
  return (int)out;
}

int floordiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
int mod(int a, int b) { return a - b * floordiv(a, b); }

/* The cell in lattice units, and its periodic wrap. */
struct sg_frame {
  int nd;
  int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
  bool periodic[3] = {false, false, false}, metallic[3] = {false, false, false};
  std::complex<double> eik[3] = {1.0, 1.0, 1.0};

  explicit sg_frame(const fields &f) : nd(f.gv.dim == D3 ? 3 : 2) {
    const ivec l = f.user_volume.little_corner(), h = f.user_volume.big_corner();
    for (int d = 0; d < nd; ++d) {
      lo[d] = l.in_direction(direction(d));
      hi[d] = h.in_direction(direction(d));
      periodic[d] = f.boundaries[High][d] == Periodic;
      metallic[d] = f.boundaries[High][d] == Metallic;
      if (periodic[d]) eik[d] = f.eikna[d];
      if (!periodic[d] && !metallic[d])
        meep::abort("conforming rows: only periodic and metallic boundaries are supported");
    }
  }
  int wrap(int d, int v) const {
    if (!periodic[d]) return v;
    const int P = hi[d] - lo[d];
    return lo[d] + mod(v - lo[d], P);
  }
  sg_key key(component c, const sg_pt &p) const {
    return {(int)c, wrap(0, p[0]), wrap(1, p[1]), wrap(2, p[2])};
  }
  // meep's Bloch condition: the field at p + n L is eik^n times that at p
  std::complex<double> phase(const sg_pt &p) const {
    std::complex<double> ph = 1.0;
    for (int d = 0; d < nd; ++d)
      if (periodic[d] && eik[d] != 1.0) {
        const int P = hi[d] - lo[d], o = p[d] - lo[d];
        ph *= std::pow(eik[d], o >= 0 ? o / P : -((P - 1 - o) / P));
      }
    return ph;
  }
  sg_pt of(const ivec &v) const {
    sg_pt p = {0, 0, 0};
    for (int d = 0; d < nd; ++d)
      p[d] = v.in_direction(direction(d));
    return p;
  }
};

void index_storage(fields &f, const sg_frame &fr, map<sg_key, vector<sg_loc> > &at) {
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    const grid_volume &gv = f.chunks[i]->gv;
    for (component c : all_components)
      if (gv.has_field(c)) LOOP_OVER_VOL(gv, c, n) {
          IVEC_LOOP_ILOC(gv, here);
          vector<sg_loc> &v = at[fr.key(c, fr.of(here))];
          sg_loc l{i, c, n, here, gv.owns(here), 0};
          l.ph = fr.phase(fr.of(here));
          /* owned first; then a ghost inside its chunk's extent, whose PML
             arrays reach it -- a staggered component also has ghosts half a
             cell beyond the upper corner, where they do not */
          auto rank = [&](const sg_loc &m) {
            if (m.owned) return 0;
            const grid_volume &mg = f.chunks[m.chunk]->gv;
            bool in = true;
            LOOP_OVER_DIRECTIONS(mg.dim, d) {
              const int x = m.here.in_direction(d);
              in = in && x >= mg.little_corner().in_direction(d) &&
                   x <= mg.big_corner().in_direction(d);
            }
            return in ? 1 : 2;
          };
          l.rank = (char)rank(l);
          auto it = v.begin();
          while (it != v.end() && it->rank <= l.rank)
            ++it;
          v.insert(it, l);
        }
  }
}

/* Refuse what the rows do not model yet, wherever they apply. */
void check_supported(fields &f, const sg_frame &fr, const set<sg_key> &touched) {
  if (f.beta != 0) meep::abort("conforming rows: beta != 0 is not supported yet");
  for (double k : f.bfast_scaled_k)
    if (k != 0) meep::abort("conforming rows: bfast is not supported yet");
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    const structure_chunk *s = f.chunks[i]->s;
    const grid_volume &gv = f.chunks[i]->gv;
    for (component c : all_components)
      if (gv.has_field(c)) LOOP_OVER_VOL(gv, c, n) {
          IVEC_LOOP_ILOC(gv, here);
          if (!touched.count(fr.key(c, fr.of(here)))) continue;
          FOR_DIRECTIONS(d) {
            const realnum *chi = s->chi1inv[c][d];
            if (d != component_direction(c) && chi && chi[n] != 0)
              meep::abort(is_magnetic(c) ? "conforming rows: no magnetic anisotropy at the "
                                           "interface yet"
                                         : "bug: electric anisotropy left at the interface");
          }
        }
  }
}

/* Per axis, a component sits on the nodes or halfway between them: E along d
   is edge-like in d, H with normal n is edge-like in the other two. */
bool node_axis(component c, int a) {
  const int d = (int)component_direction(c);
  return is_electric(c) ? a != d : a == d;
}

/* The refined region, the two lattices, and the grid they make together. */
struct sg_grid {
  const sg_frame &fr;
  int nd, S;                    // S: coarse node spacing in lattice units; fine is 2
  sg_pt oc, of;                 // lattice origins of coarse and fine nodes
  vector<array<int, 6> > boxes; // fine chunks' node boxes: lo x y z, hi x y z

  sg_grid(const fields &f, const sg_frame &frame, int qc)
      : fr(frame), nd(frame.nd), S(2 * qc), oc{0, 0, 0}, of{0, 0, 0} {
    bool have_c = false, have_f = false;
    for (int i = 0; i < f.num_chunks; ++i) {
      const grid_volume &gv = f.chunks[i]->gv;
      const sg_pt l = fr.of(gv.little_corner()), b = fr.of(gv.big_corner());
      if (gv.q() == qc) {
        for (int d = 0; d < nd; ++d)
          oc[d] = mod(l[d], S);
        have_c = true;
      }
      else {
        boxes.push_back({l[0], l[1], l[2], b[0], b[1], b[2]});
        for (int d = 0; d < nd; ++d)
          of[d] = mod(l[d], 2);
        have_f = true;
      }
    }
    if (!have_c || !have_f) meep::abort("conforming rows: need a coarse and a fine chunk");
  }

  // is the unit cube with lower corner a in the fine region
  bool cube_in(const sg_pt &a) const {
    int im[3][3], n[3];
    for (int d = 0; d < 3; ++d) {
      n[d] = 1, im[d][0] = a[d];
      if (d < nd && fr.periodic[d]) {
        const int P = fr.hi[d] - fr.lo[d];
        n[d] = 3, im[d][1] = a[d] - P, im[d][2] = a[d] + P;
      }
    }
    for (const array<int, 6> &b : boxes)
      for (int i = 0; i < n[0]; ++i)
        for (int j = 0; j < n[1]; ++j)
          for (int k = 0; k < n[2]; ++k) {
            const int p[3] = {im[0][i], im[1][j], im[2][k]};
            bool in = true;
            for (int d = 0; d < nd && in; ++d)
              in = b[d] <= p[d] && p[d] + 1 <= b[d + 3];
            if (in) return true;
          }
    return false;
  }
  bool beyond_wall(const sg_pt &a) const {
    for (int d = 0; d < nd; ++d)
      if (fr.metallic[d] && (a[d] < fr.lo[d] || a[d] + 1 > fr.hi[d])) return true;
    return false;
  }
  // every unit cube in [lo, hi] (inclusive corners) inside: 2, some: 1, none: 0
  int inside(const sg_pt &lo, const sg_pt &hi) const {
    int in = 0, n = 0;
    for (int x = lo[0]; x <= hi[0]; ++x)
      for (int y = lo[1]; y <= hi[1]; ++y)
        for (int z = lo[2]; z <= hi[2]; ++z) {
          const sg_pt a = {x, y, z};
          if (beyond_wall(a)) continue;
          ++n;
          in += cube_in(a);
        }
    return in == n ? 2 : in ? 1 : 0;
  }
  // within R lattice units of the interface: the cubes around p are mixed
  bool near_interface(const sg_pt &p, int R) const {
    sg_pt lo = p, hi = p;
    for (int d = 0; d < nd; ++d)
      lo[d] = p[d] - R - 1, hi[d] = p[d] + R;
    return inside(lo, hi) == 1;
  }
  bool coarse(int d, int v) const { return d >= nd || mod(v - oc[d], S) == 0; }
  bool fine(int d, int v) const { return d >= nd || mod(v - of[d], 2) == 0; }
  bool on_wall(int d, int v) const {
    return d < nd && fr.metallic[d] && (v <= fr.lo[d] || v >= fr.hi[d]);
  }

  /* A sample: component c whose lower node is q, spanning s along its
     edge-like axes (2 fine, S coarse).  Its key is where meep stores it. */
  sg_key key(component c, const sg_pt &q, int s) const {
    sg_pt p = q;
    for (int a = 0; a < nd; ++a)
      if (!node_axis(c, a)) p[a] += s / 2;
    return fr.key(c, p);
  }
  bool has_edge_axis(component c) const {
    for (int a = 0; a < nd; ++a)
      if (!node_axis(c, a)) return true;
    return false;
  }
  // E tangential, or H normal, to a metal wall is zero
  bool zero(component c, const sg_pt &q) const {
    for (int a = 0; a < nd; ++a)
      if (node_axis(c, a) && on_wall(a, q[a])) return true;
    return false;
  }
  enum status { invalid, dof, slave, null };
  status classify(component c, const sg_pt &q, int s) const {
    if (zero(c, q)) return null;
    bool on_c = true, on_f = true;
    sg_pt lo = q, hi = q;
    for (int a = 0; a < nd; ++a) {
      on_c = on_c && coarse(a, q[a]), on_f = on_f && fine(a, q[a]);
      if (node_axis(c, a))
        lo[a] = q[a] - 1, hi[a] = q[a];
      else
        hi[a] = q[a] + s - 1;
    }
    // a sample with no edge-like axis (Ez in 2-D) is coarse if its node is
    const bool is_c = has_edge_axis(c) ? s == S && on_c : on_c;
    if (!(has_edge_axis(c) ? (s == S ? on_c : on_f) : on_f)) return invalid;
    switch (inside(lo, hi)) {
      case 2: return has_edge_axis(c) && s == S ? invalid : dof;
      case 1: return is_c ? dof : slave;
      default: return is_c ? dof : invalid;
    }
  }
};

/* The native rows, after the prototype's Box3D, for any number of fine boxes:
   one grid of edges and faces, rasterised exactly on the lattice (whose unit
   cubes are the prototype's h/2 voxels), H = C E, E = -W_E^-1 C^T W_H H. */
/* meep couples E_c to D_d (c != d) through the node at each cell's lower
   corner, with weight chi1inv[c][d] at that cell.  This is energy-conserving
   only if the weights are symmetric, chi1inv[c][d] == chi1inv[d][c], and none
   reaches across the interface; so symmetrise every node and drop the
   off-diagonal terms of nodes within a coarse cell of the interface.  Changes
   the structure these fields share. */
void condition_anisotropy(fields &f, const sg_frame &fr, const sg_grid &g) {
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    structure_chunk *s = f.chunks[i]->s;
    const grid_volume &gv = f.chunks[i]->gv;
    const size_t nt = gv.ntot();
    for (int a = 0; a < fr.nd; ++a)
      for (int b = a + 1; b < fr.nd; ++b) {
        const component ca = direction_component(Ex, direction(a)),
                        cb = direction_component(Ex, direction(b));
        realnum *&uab = s->chi1inv[ca][b], *&uba = s->chi1inv[cb][a];
        if (!uab && !uba) continue;
        for (int e = 0; e < 2; ++e) {
          const component c = e ? cb : ca;
          realnum *&u = e ? uba : uab, *&diag = s->chi1inv[c][component_direction(c)];
          if (u) continue;
          u = new realnum[nt];
          std::fill(u, u + nt, 0);
          if (!diag) { // meep reads a missing diagonal as 1, but only with no off-diagonal
            diag = new realnum[nt];
            std::fill(diag, diag + nt, 1);
          }
        }
        const bool trivial = s->trivial_chi1inv[ca][b] && s->trivial_chi1inv[cb][a];
        s->trivial_chi1inv[ca][b] = s->trivial_chi1inv[cb][a] = trivial;
        for (size_t n = 0; n < nt; ++n)
          uab[n] = uba[n] = (realnum)(0.5 * (uab[n] + uba[n]));
      }
    FOR_ELECTRIC_COMPONENTS(c) {
      if (!gv.has_field(c)) continue;
      const direction dc = component_direction(c);
      LOOP_OVER_VOL(gv, c, n) {
        bool any = false;
        for (int d = 0; d < fr.nd; ++d)
          any = any || (direction(d) != dc && s->chi1inv[c][d] && s->chi1inv[c][d][n] != 0);
        if (!any) continue;
        IVEC_LOOP_ILOC(gv, here);
        ivec node = here;
        node.set_direction(dc, here.in_direction(dc) - gv.q());
        if (!g.near_interface(fr.of(node), g.S)) continue;
        for (int d = 0; d < fr.nd; ++d)
          if (direction(d) != dc && s->chi1inv[c][d]) s->chi1inv[c][d][n] = 0;
      }
    }
  }
}

conforming_rows *build_rows(fields &f) {
  int qc = 0, qf = 1 << 30;
  for (int i = 0; i < f.num_chunks; ++i) {
    qc = std::max(qc, f.chunks[i]->gv.q());
    qf = std::min(qf, f.chunks[i]->gv.q());
  }
  conforming_rows *o = new conforming_rows;
  o->ncmp = f.is_real ? 1 : 2;
  o->local_components = component_signature(f);
  o->components = global_signature(o->local_components);
  for (int i = 0; i < f.num_chunks; ++i)
    o->layout.push_back(f.chunks[i]);
  if (qc == qf) return o;
  if (f.gv.dim != D2 && f.gv.dim != D3) meep::abort("conforming rows: 2-D and 3-D only");
  if (qf != 1) meep::abort("conforming rows: the finest chunks must have q = 1");
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->gv.q() != qc && f.chunks[i]->gv.q() != qf)
      meep::abort("conforming rows: only two resolutions so far");

  const sg_frame fr(f);
  const sg_grid g(f, fr, qc);
  for (int d = 0; d < fr.nd; ++d)
    o->bloch = o->bloch || fr.eik[d] != 1.0;
  if (o->bloch && o->ncmp < 2) meep::abort("conforming rows: Bloch phases need complex fields");
  const int nd = g.nd, S = g.S;
  const int band = 0, margin = 2 * S;             // rows at the interface only
  const double unit = 0.5 * f.gv.inva / f.gv.q(); // lattice unit in meep lengths

  // the components this simulation steps: in 2-D, TM and TE as whole sets
  vector<component> ecomp, hcomp;
  {
    const int sig = o->components;
    auto any = [&](std::initializer_list<component> cs) {
      for (component c : cs)
        if (sig & (1 << c)) return true;
      return false;
    };
    const bool tm = nd == 3 ? any({Ex, Ey, Ez, Hx, Hy, Hz}) : any({Ez, Hx, Hy});
    const bool te = nd == 3 ? tm : any({Ex, Ey, Hz});
    if (nd == 3 && tm)
      ecomp = {Ex, Ey, Ez}, hcomp = {Hx, Hy, Hz};
    else {
      if (tm) ecomp.push_back(Ez), hcomp.push_back(Hx), hcomp.push_back(Hy);
      if (te) ecomp.push_back(Ex), ecomp.push_back(Ey), hcomp.push_back(Hz);
    }
  }

  // --- rasterise: each unit cube goes to one sample per component
  struct sample {
    sg_pt q;
    int s;
  };
  map<sg_key, double> w;
  map<sg_key, sample> hsamples;
  auto pos = [](const sg_key &k) { return sg_pt{k[1], k[2], k[3]}; };
  auto wrapped = [&](const sg_pt &p) {
    return sg_pt{fr.wrap(0, p[0]), fr.wrap(1, p[1]), fr.wrap(2, p[2])};
  };
  auto owner = [&](bool fine, int a, int v, bool lower) {
    const int s = fine ? 2 : S, org = fine ? g.of[a] : g.oc[a];
    return org + s * floordiv(v - org + (lower ? 0 : s / 2), s);
  };
  auto raster = [&](const sg_pt &px, component c) {
    sg_pt lo = px, hi = px;
    for (int a = 0; a < nd; ++a)
      if (node_axis(c, a)) lo[a] = px[a] - 1, hi[a] = px[a] + 1;
    const bool fine = g.inside(lo, hi) == 2;
    sg_pt q = {0, 0, 0};
    for (int a = 0; a < nd; ++a)
      q[a] = owner(fine, a, px[a], !node_axis(c, a));
    if (g.zero(c, q)) return;
    const int s = fine ? 2 : S;
    const sg_key k = g.key(c, q, s);
    w[k] += 1;
    if (is_magnetic(c)) hsamples[k] = sample{wrapped(q), s};
  };
  {
    set<sg_pt> seen;
    for (const array<int, 6> &b : g.boxes) {
      int lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
      for (int d = 0; d < nd; ++d) {
        lo[d] = b[d] - margin, hi[d] = b[d + 3] + margin;
        if (fr.periodic[d])
          hi[d] = std::min(hi[d], lo[d] + fr.hi[d] - fr.lo[d]);
        else
          lo[d] = std::max(lo[d], fr.lo[d]), hi[d] = std::min(hi[d], fr.hi[d]);
      }
      for (int x = lo[0]; x < hi[0]; ++x)
        for (int y = lo[1]; y < hi[1]; ++y)
          for (int z = lo[2]; z < hi[2]; ++z) {
            const sg_pt px = {fr.wrap(0, x), fr.wrap(1, y), nd == 3 ? fr.wrap(2, z) : 0};
            if (!seen.insert(px).second) continue;
            for (component c : ecomp)
              raster(px, c);
            for (component c : hcomp)
              raster(px, c);
          }
    }
  }

  // --- dofs, numbered as the rows reach them
  map<sg_key, int> index;
  vector<sg_key> keys;
  auto dof = [&](const sg_key &k) {
    auto it = index.find(k);
    if (it != index.end()) return it->second;
    index[k] = (int)keys.size();
    keys.push_back(k);
    return (int)keys.size() - 1;
  };
  map<sg_key, vector<sg_term> > slaves;
  /* An E sample as dofs: itself, nothing on a wall, or, on the interface, the
     coarse samples it is slaved to -- replicated along its own edge-like axis,
     linear along a node-like one that is off the coarse lattice. */
  function<vector<sg_term>(component, sg_pt, int)> resolve = [&](component c, sg_pt q,
                                                                 int s) -> vector<sg_term> {
    const std::complex<double> ph0 = fr.phase(q); // q may lie beyond a periodic boundary
    q = wrapped(q);
    switch (g.classify(c, q, s)) {
      case sg_grid::null: return {};
      case sg_grid::dof: return {sg_term{dof(g.key(c, q, s)), 1.0, ph0}};
      case sg_grid::invalid:
        meep::abort("conforming rows: component %d at (%d,%d,%d) is not on the grid", (int)c, q[0],
                    q[1], q[2]);
      case sg_grid::slave: break;
    }
    sg_pt base = q;
    vector<int> free;
    for (int a = 0; a < nd; ++a) {
      if (!node_axis(c, a))
        base[a] = q[a] - mod(q[a] - g.oc[a], S);
      else if (!g.coarse(a, q[a])) {
        base[a] = q[a] - mod(q[a] - g.oc[a], S);
        free.push_back(a);
      }
    }
    vector<sg_term> out;
    for (int bits = 0; bits < (1 << free.size()); ++bits) {
      sg_pt m = base;
      double wt = 1.0;
      for (size_t b = 0; b < free.size(); ++b) {
        const int a = free[b];
        const double frac = double(q[a] - base[a]) / S;
        if (bits >> b & 1)
          m[a] += S, wt *= frac;
        else
          wt *= 1.0 - frac;
      }
      if (wt == 0.0) continue;
      const sg_grid::status st = g.classify(c, wrapped(m), S);
      if (st != sg_grid::dof && st != sg_grid::null)
        meep::abort("conforming rows: a slave's master is not a dof");
      for (const sg_term &t : resolve(c, m, S))
        out.push_back(sg_term{t.k, wt * t.w, t.ph});
    }
    slaves[g.key(c, q, s)] = out;
    for (sg_term &t : out)
      t.ph *= ph0;
    return out;
  };

  // --- C: dH_n/dt = -(dE_b/da - dE_a/db), (n, a, b) cyclic, on each face
  map<sg_key, vector<sg_term> > C;
  map<int, vector<pair<sg_key, sg_term> > > CT; // an E dof's H rows: weight, phase
  const component E_of[3] = {Ex, Ey, Ez};
  for (const auto &kh : hsamples) {
    const component hc = component(kh.first[0]);
    const int n = (int)component_direction(hc), a = (n + 1) % 3, b = (n + 2) % 3;
    const sg_pt &q = kh.second.q;
    const int s = kh.second.s;
    const double inv = 1.0 / (s * unit);
    vector<sg_term> row;
    auto add = [&](component ec, sg_pt at, int along, double v) {
      if (along >= 0) at[along] += s;
      for (const sg_term &t : resolve(ec, at, s))
        row.push_back(sg_term{t.k, v * t.w, t.ph});
    };
    if (a < nd) add(E_of[b], q, a, -inv), add(E_of[b], q, -1, inv);
    if (b < nd) add(E_of[a], q, b, inv), add(E_of[a], q, -1, -inv);
    if (row.empty()) continue;
    for (const sg_term &t : row)
      CT[t.k].push_back({kh.first, t});
    C[kh.first] = row;
  }

  // --- the replaced rows
  for (const auto &kr : C) {
    if (!g.near_interface(pos(kr.first), band)) continue;
    o->hdof.push_back(dof(kr.first));
    o->hrow.push_back(kr.second);
  }
  for (const auto &kw : w) {
    const sg_key &k = kw.first;
    if (!is_electric(component(k[0])) || !g.near_interface(pos(k), band)) continue;
    const int kd = dof(k);
    vector<sg_term> row;
    for (const pair<sg_key, sg_term> &hc : CT[kd]) // the adjoint: conjugate phases
      row.push_back(
          sg_term{dof(hc.first), -hc.second.w * w[hc.first] / kw.second, conj(hc.second.ph)});
    o->edof.push_back(kd);
    o->erow.push_back(row);
  }

  /* Every other stored position on the interface gets the value the grid
     implies there, so that nothing meep reads or outputs is stale: a fine E
     sample by the slave rule (fine edges along a box edge, which no face of
     the grid uses, included), and a fine H face lying in the interface by its
     coarse face -- normal B is constant over a coarse face. */
  map<sg_key, vector<sg_loc> > at;
  index_storage(f, fr, at);
  set<sg_key> stored; // every chunk's positions, local or not
  for (int i = 0; i < f.num_chunks; ++i) {
    const grid_volume &gv = f.chunks[i]->gv;
    for (component c : all_components)
      if (gv.has_field(c)) LOOP_OVER_VOL(gv, c, n) {
          IVEC_LOOP_ILOC(gv, here);
          stored.insert(fr.key(c, fr.of(here)));
        }
  }
  for (const sg_key &k : stored) {
    if (index.count(k) || slaves.count(k)) continue;
    const component c = component(k[0]);
    if (!is_electric(c) && !is_magnetic(c)) continue;
    sg_pt q = pos(k);
    bool fine_sample = true;
    for (int a = 0; a < nd && fine_sample; ++a) {
      if (!node_axis(c, a)) q[a] -= 1;
      fine_sample = g.fine(a, q[a]);
    }
    if (!fine_sample || (!g.has_edge_axis(c) && !g.fine(0, q[0]))) continue;
    if (g.classify(c, q, 2) != sg_grid::slave) continue;
    if (is_electric(c)) {
      resolve(c, q, 2);
      continue;
    }
    sg_pt base = q;
    bool on_plane = true;
    for (int a = 0; a < nd; ++a)
      if (!node_axis(c, a))
        base[a] = q[a] - mod(q[a] - g.oc[a], S);
      else
        on_plane = on_plane && g.coarse(a, q[a]);
    if (!on_plane || g.classify(c, wrapped(base), S) != sg_grid::dof) continue;
    slaves[k] = {sg_term{dof(g.key(c, wrapped(base), S)), 1.0, fr.phase(base)}};
  }

  {
    set<sg_key> touched;
    for (int k : o->hdof)
      touched.insert(keys[k]);
    for (int k : o->edof)
      touched.insert(keys[k]);
    for (const auto &ks : slaves)
      touched.insert(ks.first);
    condition_anisotropy(f, fr, g);
    check_supported(f, fr, touched);
  }

  // --- where meep stores each of them
  o->where.resize(keys.size());
  for (size_t k = 0; k < keys.size(); ++k)
    o->where[k] = at[keys[k]];
  for (const auto &ks : slaves) {
    o->swhere.push_back(at[ks.first]);
    o->srow.push_back(ks.second);
    o->smagnetic.push_back(is_magnetic(component(ks.first[0])));
  }

  o->finish(f);
  o->nd = nd;
  o->vol.assign(keys.size(), 0.0);
  for (size_t k = 0; k < keys.size(); ++k) {
    auto it = w.find(keys[k]);
    if (it != w.end()) o->vol[k] = it->second * std::pow(unit, nd);
  }
  // within a coarse cell of the interface; beyond, a dof's volume is a plain cell
  for (size_t k = 0; k < keys.size(); ++k)
    if (o->vol[k] > 0 && g.near_interface(pos(keys[k]), S))
      o->src_to[keys[k]] = {sg_term{(int)k, 1.0}};
  for (const auto &ks : slaves) {
    bool ok = !ks.second.empty();
    for (const sg_term &q : ks.second)
      ok = ok && o->vol[q.k] > 0;
    if (ok) o->src_to[ks.first] = ks.second;
  }
  o->key_of = [fr](component c, const ivec &l) { return fr.key(c, fr.of(l)); };
  o->phase_of = [fr](const ivec &l) { return fr.phase(fr.of(l)); };
  o->take_sources(f);
  if (verbosity > 0)
    master_printf("conforming rows: %zu dofs, %zu H and %zu E rows, %zu slaved positions\n",
                  keys.size(), o->hdof.size(), o->edof.size(), o->swhere.size());
  return o;
}

} // namespace

void fields::free_subgrid_rows() {
  if (subgrid_rows) subgrid_rows->restore_sources(*this);
  delete subgrid_rows;
  subgrid_rows = NULL;
}

void fields::subgrid_ldos_terms(std::complex<double> &EJ, std::complex<double> &HJ,
                                double &Jsum) const {
  if (subgrid_rows) subgrid_rows->ldos_terms(*this, EJ, HJ, Jsum);
}

/* 0: start of the step; 1, 3: after B, D are stepped, before sources;
   2, 4: after H, E are derived and exchanged */
void fields::step_subgrid_rows(int stage) {
  bool refined = false; // every process holds the whole layout
  for (int i = 1; i < num_chunks && !refined; ++i)
    refined = chunks[i]->gv.q() != chunks[0]->gv.q();
  if (!refined) {
    if (subgrid_rows) free_subgrid_rows();
    return;
  }
  if (subgrid_rows && stage == 0) { // the layout changes only between steps
    bool same = (int)subgrid_rows->layout.size() == num_chunks &&
                subgrid_rows->local_components == component_signature(*this);
    for (int i = 0; same && i < num_chunks; ++i)
      same = subgrid_rows->layout[i] == chunks[i];
    if (or_to_all(!same)) free_subgrid_rows(); // rebuilding is collective
  }
  if (!subgrid_rows) {
    if (stage != 0) return; // start on a step boundary
    subgrid_rows = build_rows(*this);
  }
  conforming_rows *o = subgrid_rows;
  if (o->where.empty()) return; // nothing refined
  if (stage >= 5) {             // right after update_eh
    o->unshift(*this, stage == 5);
    return;
  }
  if (stage == 0 && or_to_all(conforming_rows::source_signature(*this) != o->src_sig))
    o->take_sources(*this); // sources added or removed since
  const bool h = stage <= 2;
  const vector<int> &dof = h ? o->hdof : o->edof;
  const vector<vector<sg_term> > &row = h ? o->hrow : o->erow;
  if (stage == 0) o->exchange(*this);
  if (stage == 2 || stage == 4)
    for (int cmp = 0; cmp < o->ncmp; ++cmp) {
      const vector<char> &derive = h ? o->hderive : o->ederive;
      for (size_t i = 0; i < dof.size(); ++i)
        if (o->site[dof[i]] && derive[i])
          o->derive(*this, o->where[dof[i]][0], cmp, (h ? o->hold : o->eold)[cmp][i],
                    (h ? o->hnew : o->enew)[cmp][i], (h ? o->hpol : o->epol)[cmp][i]);
    }
  if (stage == 2 || stage == 4) {
    o->restore_derived(*this, stage == 2);
    o->exchange(*this);
  }
  for (int cmp = 0; cmp < o->ncmp; ++cmp) {
    vector<conforming_rows::state> &old = h ? o->hold[cmp] : o->eold[cmp];
    vector<conforming_rows::state> &now = h ? o->hnew[cmp] : o->enew[cmp];
    if (stage == 0) {
      for (size_t i = 0; i < o->hdof.size(); ++i)
        if (o->site[o->hdof[i]]) o->hold[cmp][i] = o->save(*this, o->where[o->hdof[i]][0], cmp);
      for (size_t i = 0; i < o->edof.size(); ++i)
        if (o->site[o->edof[i]]) o->eold[cmp][i] = o->save(*this, o->where[o->edof[i]][0], cmp);
    }
    else if (stage == 1 || stage == 3) {
      for (size_t i = 0; i < dof.size(); ++i) {
        if (!o->site[dof[i]]) continue;
        const double acc = o->row_sum(row[i], dof[i], cmp);
        now[i] = o->step_point(*this, o->where[dof[i]][0], cmp, old[i], dt * acc);
      }
      if (cmp == o->ncmp - 1) o->apply_sources(*this, stage == 1);
    }
    else
      for (size_t i = 0; i < dof.size(); ++i)
        o->spread(*this, dof[i], cmp);
  }
  if (stage == 2) o->slaves(*this, true);
  if (stage == 4) o->slaves(*this, false);
}

} // namespace meep
