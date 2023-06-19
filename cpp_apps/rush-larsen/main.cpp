#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <fstream>

#define STATE_Xr1 0
#define STATE_Xr2 1
#define STATE_Xs 2
#define STATE_m 3
#define STATE_h 4
#define STATE_j 5
#define STATE_d 6
#define STATE_f 7
#define STATE_f2 8
#define STATE_fCass 9
#define STATE_s 10
#define STATE_r 11
#define STATE_Ca_SR 12
#define STATE_Ca_i 13
#define STATE_Ca_ss 14
#define STATE_R_prime 15
#define STATE_Na_i 16
#define STATE_V 17
#define STATE_K_i 18
#define NUM_STATES 19

#define PARAM_P_kna 0
#define PARAM_g_K1 1
#define PARAM_g_Kr 2
#define PARAM_g_Ks 3
#define PARAM_g_Na 4
#define PARAM_g_bna 5
#define PARAM_g_CaL 6
#define PARAM_g_bca 7
#define PARAM_g_to 8
#define PARAM_K_mNa 9
#define PARAM_K_mk 10
#define PARAM_P_NaK 11
#define PARAM_K_NaCa 12
#define PARAM_K_sat 13
#define PARAM_Km_Ca 14
#define PARAM_Km_Nai 15
#define PARAM_alpha 16
#define PARAM_gamma 17
#define PARAM_K_pCa 18
#define PARAM_g_pCa 19
#define PARAM_g_pK 20
#define PARAM_Buf_c 21
#define PARAM_Buf_sr 22
#define PARAM_Buf_ss 23
#define PARAM_Ca_o 24
#define PARAM_EC 25
#define PARAM_K_buf_c 26
#define PARAM_K_buf_sr 27
#define PARAM_K_buf_ss 28
#define PARAM_K_up 29
#define PARAM_V_leak 30
#define PARAM_V_rel 31
#define PARAM_V_sr 32
#define PARAM_V_ss 33
#define PARAM_V_xfer 34 
#define PARAM_Vmax_up 35 
#define PARAM_k1_prime 36
#define PARAM_k2_prime 37
#define PARAM_k3 38
#define PARAM_k4 39
#define PARAM_max_sr 40
#define PARAM_min_sr 41
#define PARAM_Na_o 42
#define PARAM_Cm 43
#define PARAM_F 44
#define PARAM_R 45
#define PARAM_T 46
#define PARAM_V_c 47
#define PARAM_stim_amplitude 48
#define PARAM_stim_duration 49
#define PARAM_stim_period 50
#define PARAM_stim_start 51
#define PARAM_K_o 52
#define NUM_PARAMS 53

void process_results(double *states, int num_states, int num_nodes){

  std::ofstream ofile;
  ofile.open("outputs.txt");
  if (ofile.is_open()) {

    ofile << "END STATES:\n";
    int count = 0;
    for (int i = 0; i < num_states*num_nodes; i += num_nodes) {
      ofile << states[i] << std::endl;
    }

    double checks[num_states];
    for (int i = 0; i < num_states; i++) {
      checks[i] = 0.0;
    }
    int n = num_nodes;
    for (int i = 0; i < n; i++) {
      checks[STATE_Xr1] += (states[n * STATE_Xr1 + i] - 0.0165);
      checks[STATE_Xr2] += (states[n * STATE_Xr2 + i] - 0.473);
      checks[STATE_Xs] += (states[n * STATE_Xs + i] - 0.0174);
      checks[STATE_m] += (states[n * STATE_m + i] - 0.00165);
      checks[STATE_h] += (states[n * STATE_h + i] - 0.749);
      checks[STATE_j] += (states[n * STATE_j + i] - 0.6788);
      checks[STATE_d] += (states[n * STATE_d + i] - 3.288e-05);
      checks[STATE_f] += (states[n * STATE_f + i] - 0.7026);
      checks[STATE_f2] += (states[n * STATE_f2 + i] - 0.9526);
      checks[STATE_fCass] += (states[n * STATE_fCass + i] - 0.9942);
      checks[STATE_s] += (states[n * STATE_s + i] - 0.999998);
      checks[STATE_r] += (states[n * STATE_r + i] - 2.347e-08);
      checks[STATE_Ca_i] += (states[n * STATE_Ca_i + i] - 0.000153);
      checks[STATE_R_prime] += (states[n * STATE_R_prime + i] - 0.8978);
      checks[STATE_Ca_SR] += (states[n * STATE_Ca_SR + i] - 4.272);
      checks[STATE_Ca_ss] += (states[n * STATE_Ca_ss + i] - 0.00042);
      checks[STATE_Na_i] += (states[n * STATE_Na_i + i] - 10.132);
      checks[STATE_V] += (states[n * STATE_V + i] - -85.423);
      checks[STATE_K_i] += (states[n * STATE_K_i + i] - 138.52);
    }
    ofile << "\n\nCHECKS\n";
    for (int i = 0; i < num_states; i++) {
      ofile << i << " " << checks[i] << " " << checks[i]/n << std::endl;
    }
    ofile.close();
  } else {
    std::cout << "Failed to create output file!" << std::endl;
  }
}

void forward_rush_larsen(double* states, double t_start,  double dt,
                         double* parameters, int n, int num_timesteps)
{
  double t = t_start;
  for (int it = 0; it < num_timesteps; it++) {
    for (int i = 0; i < n; i++) {
      // Assign states
      double Xr1,Xr2,Xs,m,h,j,d,f,f2,fCass,s,r,Ca_i,R_prime,Ca_SR,Ca_ss,Na_i,V,K_i;

      // Assign parameters
      double P_kna,g_K1,g_Kr,g_Ks,g_Na,g_bna,g_CaL,g_bca,g_to,K_mNa,K_mk,P_NaK,K_NaCa,K_sat,Km_Ca,Km_Nai,alpha,gamma,K_pCa,g_pCa,g_pK,Buf_c,Buf_sr,Buf_ss,Ca_o,EC,K_buf_c,K_buf_sr,K_buf_ss,K_up,V_leak,V_rel,V_sr,V_ss,V_xfer,Vmax_up,k1_prime,k2_prime,k3,k4,max_sr,min_sr,Na_o,Cm,F,R,T,V_c,stim_amplitude,stim_duration,stim_period,stim_start,K_o;
      
      // Assign states
      Xr1 = states[n * STATE_Xr1 + i];
      Xr2 = states[n * STATE_Xr2 + i];
      Xs = states[n * STATE_Xs + i];
      m = states[n * STATE_m + i];
      h = states[n * STATE_h + i];
      j = states[n * STATE_j + i];
      d = states[n * STATE_d + i];
      f = states[n * STATE_f + i];
      f2 = states[n * STATE_f2 + i];
      fCass = states[n * STATE_fCass + i];
      s = states[n * STATE_s + i];
      r = states[n * STATE_r + i];
      Ca_i = states[n * STATE_Ca_i + i];
      R_prime = states[n * STATE_R_prime + i];
      Ca_SR = states[n * STATE_Ca_SR + i];
      Ca_ss = states[n * STATE_Ca_ss + i];
      Na_i = states[n * STATE_Na_i + i];
      V = states[n * STATE_V + i];
      K_i = states[n * STATE_K_i + i];
       
       // Assign parameters
      P_kna = parameters[n * PARAM_P_kna + i];
      g_K1 = parameters[n * PARAM_g_K1 + i];
      g_Kr = parameters[n * PARAM_g_Kr + i];
      g_Ks = parameters[n * PARAM_g_Ks + i];
      g_Na = parameters[n * PARAM_g_Na + i];
      g_bna = parameters[n * PARAM_g_bna + i];
      g_CaL = parameters[n * PARAM_g_CaL + i];
      g_bca = parameters[n * PARAM_g_bca + i];
      g_to = parameters[n * PARAM_g_to + i];
      K_mNa = parameters[n * PARAM_K_mNa + i];
      K_mk = parameters[n * PARAM_K_mk + i];
      P_NaK = parameters[n * PARAM_P_NaK + i];
      K_NaCa = parameters[n * PARAM_K_NaCa + i];
      K_sat = parameters[n * PARAM_K_sat + i];
      Km_Ca = parameters[n * PARAM_Km_Ca + i];
      Km_Nai = parameters[n * PARAM_Km_Nai + i];
      alpha = parameters[n * PARAM_alpha + i];
      gamma = parameters[n * PARAM_gamma + i];
      K_pCa = parameters[n * PARAM_K_pCa + i];
      g_pCa = parameters[n * PARAM_g_pCa + i];
      g_pK = parameters[n * PARAM_g_pK + i];
      Buf_c = parameters[n * PARAM_Buf_c + i];
      Buf_sr = parameters[n * PARAM_Buf_sr + i];
      Buf_ss = parameters[n * PARAM_Buf_ss + i];
      Ca_o = parameters[n * PARAM_Ca_o + i];
      EC = parameters[n * PARAM_EC + i];
      K_buf_c = parameters[n * PARAM_K_buf_c + i];
      K_buf_sr = parameters[n * PARAM_K_buf_sr + i];
      K_buf_ss = parameters[n * PARAM_K_buf_ss + i];
      K_up = parameters[n * PARAM_K_up + i];
      V_leak = parameters[n * PARAM_V_leak + i];
      V_rel = parameters[n * PARAM_V_rel + i];
      V_sr = parameters[n * PARAM_V_sr + i];
      V_ss = parameters[n * PARAM_V_ss + i];
      V_xfer = parameters[n * PARAM_V_xfer + i];
      Vmax_up = parameters[n * PARAM_Vmax_up + i];
      k1_prime = parameters[n * PARAM_k1_prime + i];
      k2_prime = parameters[n * PARAM_k2_prime + i];
      k3 = parameters[n * PARAM_k3 + i];
      k4 = parameters[n * PARAM_k4 + i];
      max_sr = parameters[n * PARAM_max_sr + i];
      min_sr = parameters[n * PARAM_min_sr + i];
      Na_o = parameters[n * PARAM_Na_o + i];
      Cm = parameters[n * PARAM_Cm + i];
      F = parameters[n * PARAM_F + i];
      R = parameters[n * PARAM_R + i];
      T = parameters[n * PARAM_T + i];
      V_c = parameters[n * PARAM_V_c + i];
      stim_amplitude = parameters[n * PARAM_stim_amplitude + i];
      stim_duration = parameters[n * PARAM_stim_duration + i];
      stim_period = parameters[n * PARAM_stim_period + i];
      stim_start = parameters[n * PARAM_stim_start + i];
      K_o = parameters[n * PARAM_K_o + i];

      // Expressions for the Reversal potentials component
      double E_Na = R*T*log(Na_o/Na_i)/F;
      double E_K = R*T*log(K_o/K_i)/F;
      double E_Ks = R*T*log((K_o + Na_o*P_kna)/(P_kna*Na_i + K_i))/F;
      double E_Ca = 0.5*R*T*log(Ca_o/Ca_i)/F;

      // Expressions for the Inward rectifier potassium current component
      double alpha_K1 = 0.1/(1. + 6.14421235332821e-6*exp(0.06*V -
            0.06*E_K));
      double beta_K1 = (0.367879441171442*exp(0.1*V - 0.1*E_K) +
          3.06060402008027*exp(0.0002*V - 0.0002*E_K))/(1. + exp(0.5*E_K
            - 0.5*V));
      double xK1_inf = alpha_K1/(alpha_K1 + beta_K1);
      double i_K1 = 0.430331482911935*g_K1*sqrt(K_o)*(-E_K + V)*xK1_inf;

      // Expressions for the Rapid time dependent potassium current component
      double i_Kr = 0.430331482911935*g_Kr*sqrt(K_o)*(-E_K + V)*Xr1*Xr2;

      // Expressions for the Xr1 gate component
      double xr1_inf = 1.0/(1. + exp(-26./7. - V/7.));
      double alpha_xr1 = 450./(1. + exp(-9./2. - V/10.));
      double beta_xr1 = 6./(1. +
          13.5813245225782*exp(0.0869565217391304*V));
      double tau_xr1 = alpha_xr1*beta_xr1;
      double dXr1_dt = (-Xr1 + xr1_inf)/tau_xr1;
      double dXr1_dt_linearized = -1./tau_xr1;
      states[n * STATE_Xr1 + i] = (fabs(dXr1_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dXr1_dt_linearized))*dXr1_dt/dXr1_dt_linearized : dt*dXr1_dt)
        + Xr1;

      // Expressions for the Xr2 gate component
      double xr2_inf = 1.0/(1. + exp(11./3. + V/24.));
      double alpha_xr2 = 3./(1. + exp(-3. - V/20.));
      double beta_xr2 = 1.12/(1. + exp(-3. + V/20.));
      double tau_xr2 = alpha_xr2*beta_xr2;
      double dXr2_dt = (-Xr2 + xr2_inf)/tau_xr2;
      double dXr2_dt_linearized = -1./tau_xr2;
      states[n * STATE_Xr2 + i] = (fabs(dXr2_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dXr2_dt_linearized))*dXr2_dt/dXr2_dt_linearized : dt*dXr2_dt)
        + Xr2;

      // Expressions for the Slow time dependent potassium current component
      double i_Ks = g_Ks*(Xs*Xs)*(-E_Ks + V);

      // Expressions for the Xs gate component
      double xs_inf = 1.0/(1. + exp(-5./14. - V/14.));
      double alpha_xs = 1400./sqrt(1. + exp(5./6. - V/6.));
      double beta_xs = 1.0/(1. + exp(-7./3. + V/15.));
      double tau_xs = 80. + alpha_xs*beta_xs;
      double dXs_dt = (-Xs + xs_inf)/tau_xs;
      double dXs_dt_linearized = -1./tau_xs;
      states[n * STATE_Xs + i] = (fabs(dXs_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dXs_dt_linearized))*dXs_dt/dXs_dt_linearized : dt*dXs_dt) +
        Xs;

      // Expressions for the Fast sodium current component
      double i_Na = g_Na*(m*m*m)*(-E_Na + V)*h*j;

      // Expressions for the m gate component
      double m_inf = 1.0/((1. +
            0.00184221158116513*exp(-0.110741971207087*V))*(1. +
            0.00184221158116513*exp(-0.110741971207087*V)));
      double alpha_m = 1.0/(1. + exp(-12. - V/5.));
      double beta_m = 0.1/(1. + exp(7. + V/5.)) + 0.1/(1. +
          exp(-1./4. + V/200.));
      double tau_m = alpha_m*beta_m;
      double dm_dt = (-m + m_inf)/tau_m;
      double dm_dt_linearized = -1./tau_m;
      states[n * STATE_m + i] = (fabs(dm_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dm_dt_linearized))*dm_dt/dm_dt_linearized : dt*dm_dt) + m;

      // Expressions for the h gate component
      double h_inf = 1.0/((1. +
            15212.5932856544*exp(0.134589502018843*V))*(1. +
            15212.5932856544*exp(0.134589502018843*V)));
      double alpha_h = (V < -40. ?
          4.43126792958051e-7*exp(-0.147058823529412*V) : 0.);
      double beta_h = (V < -40. ? 310000.*exp(0.3485*V) +
          2.7*exp(0.079*V) : 0.77/(0.13 +
            0.0497581410839387*exp(-0.0900900900900901*V)));
      double tau_h = 1.0/(alpha_h + beta_h);
      double dh_dt = (-h + h_inf)/tau_h;
      double dh_dt_linearized = -1./tau_h;
      states[n * STATE_h + i] = (fabs(dh_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dh_dt_linearized))*dh_dt/dh_dt_linearized : dt*dh_dt) + h;

      // Expressions for the j gate component
      double j_inf = 1.0/((1. +
            15212.5932856544*exp(0.134589502018843*V))*(1. +
            15212.5932856544*exp(0.134589502018843*V)));
      double alpha_j = (V < -40. ? (37.78 + V)*(-25428.*exp(0.2444*V)
            - 6.948e-6*exp(-0.04391*V))/(1. + 50262745825.954*exp(0.311*V))
          : 0.);
      double beta_j = (V < -40. ? 0.02424*exp(-0.01052*V)/(1. +
            0.00396086833990426*exp(-0.1378*V)) : 0.6*exp(0.057*V)/(1. +
            0.0407622039783662*exp(-0.1*V)));
      double tau_j = 1.0/(alpha_j + beta_j);
      double dj_dt = (-j + j_inf)/tau_j;
      double dj_dt_linearized = -1./tau_j;
      states[n * STATE_j + i] = (fabs(dj_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dj_dt_linearized))*dj_dt/dj_dt_linearized : dt*dj_dt) + j;

      // Expressions for the Sodium background current component
      double i_b_Na = g_bna*(-E_Na + V);

      // Expressions for the L_type Ca current component
      double V_eff = (fabs(-15. + V) < 0.01 ? 0.01 : -15. + V);
      double i_CaL = 4.*g_CaL*(F*F)*(-Ca_o +
          0.25*Ca_ss*exp(2.*F*V_eff/(R*T)))*V_eff*d*f*f2*fCass/(R*T*(-1. +
            exp(2.*F*V_eff/(R*T))));

      // Expressions for the d gate component
      double d_inf = 1.0/(1. +
          0.344153786865412*exp(-0.133333333333333*V));
      double alpha_d = 0.25 + 1.4/(1. + exp(-35./13. - V/13.));
      double beta_d = 1.4/(1. + exp(1. + V/5.));
      double gamma_d = 1.0/(1. + exp(5./2. - V/20.));
      double tau_d = alpha_d*beta_d + gamma_d;
      double dd_dt = (-d + d_inf)/tau_d;
      double dd_dt_linearized = -1./tau_d;
      states[n * STATE_d + i] = (fabs(dd_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dd_dt_linearized))*dd_dt/dd_dt_linearized : dt*dd_dt) + d;

      // Expressions for the f gate component
      double f_inf = 1.0/(1. + exp(20./7. + V/7.));
      double tau_f = 20. + 180./(1. + exp(3. + V/10.)) + 200./(1. +
          exp(13./10. - V/10.)) + 1102.5*exp(-((27. + V)*(27. + V))/225.);
      double df_dt = (-f + f_inf)/tau_f;
      double df_dt_linearized = -1./tau_f;
      states[n * STATE_f + i] = (fabs(df_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*df_dt_linearized))*df_dt/df_dt_linearized : dt*df_dt) + f;

      // Expressions for the F2 gate component
      double f2_inf = 0.33 + 0.67/(1. + exp(5. + V/7.));
      double tau_f2 = 31./(1. + exp(5./2. - V/10.)) + 80./(1. +
          exp(3. + V/10.)) + 562.*exp(-((27. + V)*(27. + V))/240.);
      double df2_dt = (-f2 + f2_inf)/tau_f2;
      double df2_dt_linearized = -1./tau_f2;
      states[n * STATE_f2 + i] = (fabs(df2_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*df2_dt_linearized))*df2_dt/df2_dt_linearized : dt*df2_dt) +
        f2;

      // Expressions for the FCass gate component
      double fCass_inf = 0.4 + 0.6/(1. + 400.0*(Ca_ss*Ca_ss));
      double tau_fCass = 2. + 80./(1. + 400.0*(Ca_ss*Ca_ss));
      double dfCass_dt = (-fCass + fCass_inf)/tau_fCass;
      double dfCass_dt_linearized = -1./tau_fCass;
      states[n * STATE_fCass + i] = (fabs(dfCass_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dfCass_dt_linearized))*dfCass_dt/dfCass_dt_linearized :
          dt*dfCass_dt) + fCass;

      // Expressions for the Calcium background current component
      double i_b_Ca = g_bca*(-E_Ca + V);

      // Expressions for the Transient outward current component
      double i_to = g_to*(-E_K + V)*r*s;

      // Expressions for the s gate component
      double s_inf = 1.0/(1. + exp(4. + V/5.));
      double tau_s = 3. + 5./(1. + exp(-4. + V/5.)) +
        85.*exp(-((45. + V)*(45. + V))/320.);
      double ds_dt = (-s + s_inf)/tau_s;
      double ds_dt_linearized = -1./tau_s;
      states[n * STATE_s + i] = (fabs(ds_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*ds_dt_linearized))*ds_dt/ds_dt_linearized : dt*ds_dt) + s;

      // Expressions for the r gate component
      double r_inf = 1.0/(1. + exp(10./3. - V/6.));
      double tau_r = 0.8 + 9.5*exp(-((40. + V)*(40. + V))/1800.);
      double dr_dt = (-r + r_inf)/tau_r;
      double dr_dt_linearized = -1./tau_r;
      states[n * STATE_r + i] = (fabs(dr_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dr_dt_linearized))*dr_dt/dr_dt_linearized : dt*dr_dt) + r;

      // Expressions for the Sodium potassium pump current component
      double i_NaK = K_o*P_NaK*Na_i/((K_mNa + Na_i)*(K_mk + K_o)*(1. +
            0.0353*exp(-F*V/(R*T)) + 0.1245*exp(-0.1*F*V/(R*T))));

      // Expressions for the Sodium calcium exchanger current component
      double i_NaCa =
        K_NaCa*(Ca_o*(Na_i*Na_i*Na_i)*exp(F*gamma*V/(R*T)) -
            alpha*(Na_o*Na_o*Na_o)*Ca_i*exp(F*(-1. + gamma)*V/(R*T)))/((1. +
              K_sat*exp(F*(-1. + gamma)*V/(R*T)))*(Ca_o +
              Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o)));

      // Expressions for the Calcium pump current component
      double i_p_Ca = g_pCa*Ca_i/(K_pCa + Ca_i);

      // Expressions for the Potassium pump current component
      double i_p_K = g_pK*(-E_K + V)/(1. +
          65.4052157419383*exp(-0.167224080267559*V));

      // Expressions for the Calcium dynamics component
      double i_up = Vmax_up/(1. + (K_up*K_up)/(Ca_i*Ca_i));
      double i_leak = V_leak*(-Ca_i + Ca_SR);
      double i_xfer = V_xfer*(-Ca_i + Ca_ss);
      double kcasr = max_sr - (max_sr - min_sr)/(1. + (EC*EC)/(Ca_SR*Ca_SR));
      double Ca_i_bufc = 1.0/(1. + Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c
              + Ca_i)));
      double Ca_sr_bufsr = 1.0/(1. + Buf_sr*K_buf_sr/((K_buf_sr +
              Ca_SR)*(K_buf_sr + Ca_SR)));
      double Ca_ss_bufss = 1.0/(1. + Buf_ss*K_buf_ss/((K_buf_ss +
              Ca_ss)*(K_buf_ss + Ca_ss)));
      double dCa_i_dt = (V_sr*(-i_up + i_leak)/V_c - Cm*(-2.*i_NaCa +
            i_b_Ca + i_p_Ca)/(2.*F*V_c) + i_xfer)*Ca_i_bufc;
      double dCa_i_bufc_dCa_i = 2.*Buf_c*K_buf_c/(((1. +
              Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i)))*(1. +
              Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i))))*((K_buf_c +
              Ca_i)*(K_buf_c + Ca_i)*(K_buf_c + Ca_i)));
      double di_NaCa_dCa_i = -K_NaCa*alpha*(Na_o*Na_o*Na_o)*exp(F*(-1.
            + gamma)*V/(R*T))/((1. + K_sat*exp(F*(-1. + gamma)*V/(R*T)))*(Ca_o +
              Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o)));
      double di_up_dCa_i = 2.*Vmax_up*(K_up*K_up)/(((1. +
              (K_up*K_up)/(Ca_i*Ca_i))*(1. +
              (K_up*K_up)/(Ca_i*Ca_i)))*(Ca_i*Ca_i*Ca_i));
      double di_p_Ca_dCa_i = g_pCa/(K_pCa + Ca_i) - g_pCa*Ca_i/((K_pCa +
            Ca_i)*(K_pCa + Ca_i));
      double dE_Ca_dCa_i = -0.5*R*T/(F*Ca_i);
      double dCa_i_dt_linearized = (-V_xfer + V_sr*(-V_leak -
            di_up_dCa_i)/V_c - Cm*(-2.*di_NaCa_dCa_i - g_bca*dE_Ca_dCa_i +
              di_p_Ca_dCa_i)/(2.*F*V_c))*Ca_i_bufc + (V_sr*(-i_up + i_leak)/V_c -
              Cm*(-2.*i_NaCa + i_b_Ca + i_p_Ca)/(2.*F*V_c) + i_xfer)*dCa_i_bufc_dCa_i;
      states[n * STATE_Ca_i + i] = Ca_i + (fabs(dCa_i_dt_linearized) > 1.0e-8 ?
          (-1.0 + exp(dt*dCa_i_dt_linearized))*dCa_i_dt/dCa_i_dt_linearized :
          dt*dCa_i_dt);
      double k1 = k1_prime/kcasr;
      double k2 = k2_prime*kcasr;
      double O = (Ca_ss*Ca_ss)*R_prime*k1/(k3 + (Ca_ss*Ca_ss)*k1);
      double dR_prime_dt = k4*(1. - R_prime) - Ca_ss*R_prime*k2;
      double dR_prime_dt_linearized = -k4 - Ca_ss*k2;
      states[n * STATE_R_prime + i] = (fabs(dR_prime_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dR_prime_dt_linearized))*dR_prime_dt/dR_prime_dt_linearized :
          dt*dR_prime_dt) + R_prime;
      double i_rel = V_rel*(-Ca_ss + Ca_SR)*O;
      double dCa_SR_dt = (-i_leak - i_rel + i_up)*Ca_sr_bufsr;
      double dkcasr_dCa_SR = -2.*(EC*EC)*(max_sr - min_sr)/(((1. +
              (EC*EC)/(Ca_SR*Ca_SR))*(1. + (EC*EC)/(Ca_SR*Ca_SR)))*(Ca_SR*Ca_SR*Ca_SR));
      double dCa_sr_bufsr_dCa_SR = 2.*Buf_sr*K_buf_sr/(((1. +
              Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR)))*(1. +
              Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR))))*((K_buf_sr +
              Ca_SR)*(K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR)));
      double di_rel_dO = V_rel*(-Ca_ss + Ca_SR);
      double dk1_dkcasr = -k1_prime/(kcasr*kcasr);
      double dO_dk1 = (Ca_ss*Ca_ss)*R_prime/(k3 + (Ca_ss*Ca_ss)*k1) -
        pow(Ca_ss, 4.)*R_prime*k1/((k3 + (Ca_ss*Ca_ss)*k1)*(k3 +
              (Ca_ss*Ca_ss)*k1));
      double di_rel_dCa_SR = V_rel*O + V_rel*(-Ca_ss +
          Ca_SR)*dO_dk1*dk1_dkcasr*dkcasr_dCa_SR;
      double dCa_SR_dt_linearized = (-V_leak - di_rel_dCa_SR -
          dO_dk1*di_rel_dO*dk1_dkcasr*dkcasr_dCa_SR)*Ca_sr_bufsr + (-i_leak - i_rel
            + i_up)*dCa_sr_bufsr_dCa_SR;
      states[n * STATE_Ca_SR + i] = Ca_SR + (fabs(dCa_SR_dt_linearized) > 1.0e-8 ?
          (-1.0 + exp(dt*dCa_SR_dt_linearized))*dCa_SR_dt/dCa_SR_dt_linearized
          : dt*dCa_SR_dt);
      double dCa_ss_dt = (V_sr*i_rel/V_ss - V_c*i_xfer/V_ss -
          Cm*i_CaL/(2.*F*V_ss))*Ca_ss_bufss;
      double dO_dCa_ss = -2.*(Ca_ss*Ca_ss*Ca_ss)*(k1*k1)*R_prime/((k3 +
            (Ca_ss*Ca_ss)*k1)*(k3 + (Ca_ss*Ca_ss)*k1)) + 2.*Ca_ss*R_prime*k1/(k3 +
            (Ca_ss*Ca_ss)*k1);
      double di_rel_dCa_ss = -V_rel*O + V_rel*(-Ca_ss + Ca_SR)*dO_dCa_ss;
      double dCa_ss_bufss_dCa_ss = 2.*Buf_ss*K_buf_ss/(((1. +
              Buf_ss*K_buf_ss/((K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss)))*(1. +
              Buf_ss*K_buf_ss/((K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss))))*((K_buf_ss +
              Ca_ss)*(K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss)));
      double di_CaL_dCa_ss =
        1.0*g_CaL*(F*F)*V_eff*d*exp(2.*F*V_eff/(R*T))*f*f2*fCass/(R*T*(-1. +
              exp(2.*F*V_eff/(R*T))));
      double dCa_ss_dt_linearized = (V_sr*(dO_dCa_ss*di_rel_dO +
            di_rel_dCa_ss)/V_ss - V_c*V_xfer/V_ss -
          Cm*di_CaL_dCa_ss/(2.*F*V_ss))*Ca_ss_bufss + (V_sr*i_rel/V_ss -
          V_c*i_xfer/V_ss - Cm*i_CaL/(2.*F*V_ss))*dCa_ss_bufss_dCa_ss;
      states[n * STATE_Ca_ss + i] = Ca_ss + (fabs(dCa_ss_dt_linearized) > 1.0e-8 ?
          (-1.0 + exp(dt*dCa_ss_dt_linearized))*dCa_ss_dt/dCa_ss_dt_linearized
          : dt*dCa_ss_dt);

      // Expressions for the Sodium dynamics component
      double dNa_i_dt = Cm*(-i_Na - i_b_Na - 3.*i_NaCa - 3.*i_NaK)/(F*V_c);
      double dE_Na_dNa_i = -R*T/(F*Na_i);
      double di_NaCa_dNa_i =
        3.*Ca_o*K_NaCa*(Na_i*Na_i)*exp(F*gamma*V/(R*T))/((1. +
              K_sat*exp(F*(-1. + gamma)*V/(R*T)))*(Ca_o +
              Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o)));
      double di_Na_dE_Na = -g_Na*(m*m*m)*h*j;
      double di_NaK_dNa_i = K_o*P_NaK/((K_mNa + Na_i)*(K_mk + K_o)*(1. +
            0.0353*exp(-F*V/(R*T)) + 0.1245*exp(-0.1*F*V/(R*T)))) -
        K_o*P_NaK*Na_i/(((K_mNa + Na_i)*(K_mNa + Na_i))*(K_mk + K_o)*(1. +
              0.0353*exp(-F*V/(R*T)) + 0.1245*exp(-0.1*F*V/(R*T))));
      double dNa_i_dt_linearized = Cm*(-3.*di_NaCa_dNa_i - 3.*di_NaK_dNa_i
          + g_bna*dE_Na_dNa_i - dE_Na_dNa_i*di_Na_dE_Na)/(F*V_c);
      states[n * STATE_Na_i + i] = Na_i + (fabs(dNa_i_dt_linearized) > 1.0e-8 ?
          (-1.0 + exp(dt*dNa_i_dt_linearized))*dNa_i_dt/dNa_i_dt_linearized :
          dt*dNa_i_dt);

      // Expressions for the Membrane component
      double i_Stim = (t - stim_period*floor(t/stim_period) <=
          stim_duration + stim_start && t - stim_period*floor(t/stim_period)
          >= stim_start ? -stim_amplitude : 0.);
      double dV_dt = -i_CaL - i_K1 - i_Kr - i_Ks - i_Na - i_NaCa - i_NaK -
        i_Stim - i_b_Ca - i_b_Na - i_p_Ca - i_p_K - i_to;
      double dalpha_K1_dV = -3.68652741199693e-8*exp(0.06*V -
          0.06*E_K)/((1. + 6.14421235332821e-6*exp(0.06*V - 0.06*E_K))*(1. +
              6.14421235332821e-6*exp(0.06*V - 0.06*E_K)));
      double di_CaL_dV_eff = 4.*g_CaL*(F*F)*(-Ca_o +
          0.25*Ca_ss*exp(2.*F*V_eff/(R*T)))*d*f*f2*fCass/(R*T*(-1. +
            exp(2.*F*V_eff/(R*T)))) - 8.*g_CaL*(F*F*F)*(-Ca_o +
          0.25*Ca_ss*exp(2.*F*V_eff/(R*T)))*V_eff*d*exp(2.*F*V_eff/(R*T))*f*f2*fCass/((R*R)*(T*T)*((-1.
              + exp(2.*F*V_eff/(R*T)))*(-1. + exp(2.*F*V_eff/(R*T))))) +
          2.0*g_CaL*(F*F*F)*Ca_ss*V_eff*d*exp(2.*F*V_eff/(R*T))*f*f2*fCass/((R*R)*(T*T)*(-1.
                + exp(2.*F*V_eff/(R*T))));
      double di_Ks_dV = g_Ks*(Xs*Xs);
      double di_p_K_dV = g_pK/(1. +
          65.4052157419383*exp(-0.167224080267559*V)) +
        10.9373270471469*g_pK*(-E_K + V)*exp(-0.167224080267559*V)/((1. +
              65.4052157419383*exp(-0.167224080267559*V))*(1. +
              65.4052157419383*exp(-0.167224080267559*V)));
      double di_to_dV = g_to*r*s;
      double dxK1_inf_dbeta_K1 = -alpha_K1/((alpha_K1 + beta_K1)*(alpha_K1 +
            beta_K1));
      double dxK1_inf_dalpha_K1 = 1.0/(alpha_K1 + beta_K1) -
        alpha_K1/((alpha_K1 + beta_K1)*(alpha_K1 + beta_K1));
      double dbeta_K1_dV = (0.000612120804016053*exp(0.0002*V -
            0.0002*E_K) + 0.0367879441171442*exp(0.1*V - 0.1*E_K))/(1. +
            exp(0.5*E_K - 0.5*V)) + 0.5*(0.367879441171442*exp(0.1*V -
              0.1*E_K) + 3.06060402008027*exp(0.0002*V -
                0.0002*E_K))*exp(0.5*E_K - 0.5*V)/((1. + exp(0.5*E_K -
                    0.5*V))*(1. + exp(0.5*E_K - 0.5*V)));
      double di_K1_dV = 0.430331482911935*g_K1*sqrt(K_o)*xK1_inf +
        0.430331482911935*g_K1*sqrt(K_o)*(-E_K +
            V)*(dalpha_K1_dV*dxK1_inf_dalpha_K1 + dbeta_K1_dV*dxK1_inf_dbeta_K1);
      double dV_eff_dV = (fabs(-15. + V) < 0.01 ? 0. : 1.);
      double di_Na_dV = g_Na*(m*m*m)*h*j;
      double di_Kr_dV = 0.430331482911935*g_Kr*sqrt(K_o)*Xr1*Xr2;
      double di_NaK_dV = K_o*P_NaK*(0.0353*F*exp(-F*V/(R*T))/(R*T) +
          0.01245*F*exp(-0.1*F*V/(R*T))/(R*T))*Na_i/((K_mNa + Na_i)*(K_mk +
            K_o)*((1. + 0.0353*exp(-F*V/(R*T)) +
                0.1245*exp(-0.1*F*V/(R*T)))*(1. + 0.0353*exp(-F*V/(R*T)) +
                0.1245*exp(-0.1*F*V/(R*T)))));
      double di_K1_dxK1_inf = 0.430331482911935*g_K1*sqrt(K_o)*(-E_K +
          V);
      double di_NaCa_dV =
        K_NaCa*(Ca_o*F*gamma*(Na_i*Na_i*Na_i)*exp(F*gamma*V/(R*T))/(R*T) -
            F*alpha*(Na_o*Na_o*Na_o)*(-1. + gamma)*Ca_i*exp(F*(-1. +
                gamma)*V/(R*T))/(R*T))/((1. + K_sat*exp(F*(-1. +
                    gamma)*V/(R*T)))*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) +
                  (Na_o*Na_o*Na_o))) - F*K_NaCa*K_sat*(-1. +
                  gamma)*(Ca_o*(Na_i*Na_i*Na_i)*exp(F*gamma*V/(R*T)) -
                    alpha*(Na_o*Na_o*Na_o)*Ca_i*exp(F*(-1. +
                        gamma)*V/(R*T)))*exp(F*(-1. + gamma)*V/(R*T))/(R*T*((1. +
                          K_sat*exp(F*(-1. + gamma)*V/(R*T)))*(1. + K_sat*exp(F*(-1. +
                              gamma)*V/(R*T))))*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) +
                          (Na_o*Na_o*Na_o)));
      double dV_dt_linearized = -g_bca - g_bna - di_K1_dV - di_Kr_dV -
        di_Ks_dV - di_NaCa_dV - di_NaK_dV - di_Na_dV - di_p_K_dV - di_to_dV -
        (dalpha_K1_dV*dxK1_inf_dalpha_K1 +
        dbeta_K1_dV*dxK1_inf_dbeta_K1)*di_K1_dxK1_inf - dV_eff_dV*di_CaL_dV_eff;
      states[n * STATE_V + i] = (fabs(dV_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dV_dt_linearized))*dV_dt/dV_dt_linearized : dt*dV_dt) + V;

      // Expressions for the Potassium dynamics component
      double dK_i_dt = Cm*(-i_K1 - i_Kr - i_Ks - i_Stim - i_p_K - i_to +
          2.*i_NaK)/(F*V_c);
      double dE_Ks_dK_i = -R*T/(F*(P_kna*Na_i + K_i));
      double dbeta_K1_dE_K = (-0.000612120804016053*exp(0.0002*V -
            0.0002*E_K) - 0.0367879441171442*exp(0.1*V - 0.1*E_K))/(1. +
            exp(0.5*E_K - 0.5*V)) - 0.5*(0.367879441171442*exp(0.1*V -
              0.1*E_K) + 3.06060402008027*exp(0.0002*V -
                0.0002*E_K))*exp(0.5*E_K - 0.5*V)/((1. + exp(0.5*E_K -
                    0.5*V))*(1. + exp(0.5*E_K - 0.5*V)));
      double di_Kr_dE_K = -0.430331482911935*g_Kr*sqrt(K_o)*Xr1*Xr2;
      double dE_K_dK_i = -R*T/(F*K_i);
      double di_Ks_dE_Ks = -g_Ks*(Xs*Xs);
      double di_to_dE_K = -g_to*r*s;
      double dalpha_K1_dE_K = 3.68652741199693e-8*exp(0.06*V -
          0.06*E_K)/((1. + 6.14421235332821e-6*exp(0.06*V - 0.06*E_K))*(1. +
              6.14421235332821e-6*exp(0.06*V - 0.06*E_K)));
      double di_K1_dE_K = -0.430331482911935*g_K1*sqrt(K_o)*xK1_inf +
        0.430331482911935*g_K1*sqrt(K_o)*(-E_K +
            V)*(dalpha_K1_dE_K*dxK1_inf_dalpha_K1 + dbeta_K1_dE_K*dxK1_inf_dbeta_K1);
      double di_p_K_dE_K = -g_pK/(1. +
          65.4052157419383*exp(-0.167224080267559*V));
      double dK_i_dt_linearized =
        Cm*(-(dE_K_dK_i*dalpha_K1_dE_K*dxK1_inf_dalpha_K1 +
              dE_K_dK_i*dbeta_K1_dE_K*dxK1_inf_dbeta_K1)*di_K1_dxK1_inf -
            dE_K_dK_i*di_K1_dE_K - dE_K_dK_i*di_Kr_dE_K - dE_K_dK_i*di_p_K_dE_K -
            dE_K_dK_i*di_to_dE_K - dE_Ks_dK_i*di_Ks_dE_Ks)/(F*V_c);
      states[n * STATE_K_i + i] = K_i + (fabs(dK_i_dt_linearized) > 1.0e-8 ? (-1.0 +
            exp(dt*dK_i_dt_linearized))*dK_i_dt/dK_i_dt_linearized : dt*dK_i_dt);
    }
    t += dt;
  }
}

void init_state_values(double* states, int n)
{
  for (int i = 0; i < n; i++) {
    states[n * STATE_Xr1 + i] = 0.0165;
    states[n * STATE_Xr2 + i] = 0.473;
    states[n * STATE_Xs + i] = 0.0174;
    states[n * STATE_m + i] = 0.00165;
    states[n * STATE_h + i] = 0.749;
    states[n * STATE_j + i] = 0.6788;
    states[n * STATE_d + i] = 3.288e-05;
    states[n * STATE_f + i] = 0.7026;
    states[n * STATE_f2 + i] = 0.9526;
    states[n * STATE_fCass + i] = 0.9942;
    states[n * STATE_s + i] = 0.999998;
    states[n * STATE_r + i] = 2.347e-08;
    states[n * STATE_Ca_i + i] = 0.000153;
    states[n * STATE_R_prime + i] = 0.8978;
    states[n * STATE_Ca_SR + i] = 4.272;
    states[n * STATE_Ca_ss + i] = 0.00042;
    states[n * STATE_Na_i + i] = 10.132;
    states[n * STATE_V + i] = -85.423;
    states[n * STATE_K_i + i] = 138.52;
  }
}

// Default parameter values
void init_parameters_values(double* parameters, int n)
{
  for (int i = 0; i < n; i++) {
    parameters[n * PARAM_P_kna + i] = 0.03;
    parameters[n * PARAM_g_K1 + i] = 5.405;
    parameters[n * PARAM_g_Kr + i] = 0.153;
    parameters[n * PARAM_g_Ks + i] = 0.098;
    parameters[n * PARAM_g_Na + i] = 14.838;
    parameters[n * PARAM_g_bna + i] = 0.00029;
    parameters[n * PARAM_g_CaL + i] = 3.98e-05;
    parameters[n * PARAM_g_bca + i] = 0.000592;
    parameters[n * PARAM_g_to + i] = 0.294;
    parameters[n * PARAM_K_mNa + i] = 40;
    parameters[n * PARAM_K_mk + i] = 1;
    parameters[n * PARAM_P_NaK + i] = 2.724;
    parameters[n * PARAM_K_NaCa + i] = 1000;
    parameters[n * PARAM_K_sat + i] = 0.1;
    parameters[n * PARAM_Km_Ca + i] = 1.38;
    parameters[n * PARAM_Km_Nai + i] = 87.5;
    parameters[n * PARAM_alpha + i] = 2.5;
    parameters[n * PARAM_gamma + i] = 0.35;
    parameters[n * PARAM_K_pCa + i] = 0.0005;
    parameters[n * PARAM_g_pCa + i] = 0.1238;
    parameters[n * PARAM_g_pK + i] = 0.0146;
    parameters[n * PARAM_Buf_c + i] = 0.2;
    parameters[n * PARAM_Buf_sr + i] = 10;
    parameters[n * PARAM_Buf_ss + i] = 0.4;
    parameters[n * PARAM_Ca_o + i] = 2;
    parameters[n * PARAM_EC + i] = 1.5;
    parameters[n * PARAM_K_buf_c + i] = 0.001;
    parameters[n * PARAM_K_buf_sr + i] = 0.3;
    parameters[n * PARAM_K_buf_ss + i] = 0.00025;
    parameters[n * PARAM_K_up + i] = 0.00025;
    parameters[n * PARAM_V_leak + i] = 0.00036;
    parameters[n * PARAM_V_rel + i] = 0.102;
    parameters[n * PARAM_V_sr + i] = 0.001094;
    parameters[n * PARAM_V_ss + i] = 5.468e-05;
    parameters[n * PARAM_V_xfer + i] = 0.0038;
    parameters[n * PARAM_Vmax_up + i] = 0.006375;
    parameters[n * PARAM_k1_prime + i] = 0.15;
    parameters[n * PARAM_k2_prime + i] = 0.045;
    parameters[n * PARAM_k3 + i] = 0.06;
    parameters[n * PARAM_k4 + i] = 0.005;
    parameters[n * PARAM_max_sr + i] = 2.5;
    parameters[n * PARAM_min_sr + i] = 1.0;
    parameters[n * PARAM_Na_o + i] = 140;
    parameters[n * PARAM_Cm + i] = 0.185;
    parameters[n * PARAM_F + i] = 96485.3415;
    parameters[n * PARAM_R + i] = 8314.472;
    parameters[n * PARAM_T + i] = 310;
    parameters[n * PARAM_V_c + i] = 0.016404;
    parameters[n * PARAM_stim_amplitude + i] = 52;
    parameters[n * PARAM_stim_duration + i] = 1;
    parameters[n * PARAM_stim_period + i] = 1000;
    parameters[n * PARAM_stim_start + i] = 10;
    parameters[n * PARAM_K_o + i] = 5.4;
  }
}

int main(int argc, char *argv[])
{
  double t_start = 0;
  double dt = 0.02E-3;

  int num_timesteps = 1000000;
  int num_nodes = 1; 

  if (argc > 1) {
    num_timesteps = atoi(argv[1]);
    printf("num_timesteps set to %d\n", num_timesteps);

    num_nodes = atoi(argv[2]);
    printf("num_nodes set to %d\n", num_nodes);

    printf("%d states, %d params\n", NUM_STATES, NUM_PARAMS);
    if(num_timesteps <= 0 || num_nodes <= 0)
      exit(EXIT_FAILURE);
  }

  unsigned int num_states = NUM_STATES;
  size_t total_num_states = num_nodes * num_states;
  size_t states_size = total_num_states * sizeof(double);
  double *states;
  states = (double*) malloc(states_size);
  init_state_values(states, num_nodes);

  unsigned int num_parameters = NUM_PARAMS;
  size_t parameters_size = num_nodes * num_parameters * sizeof(double);
  double *parameters;
  parameters = (double*) malloc(parameters_size);
  init_parameters_values(parameters, num_nodes);

  // double *t;
  // t = (double *)malloc(1);
  // t[0] = t_start;

  struct timespec timestamp_start, timestamp_now;
  double time_elapsed;

  clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_start);
  printf("CPU: Rush Larsen (exp integrator on all gates)\n");

  forward_rush_larsen(states, t_start, dt, parameters, num_nodes, num_timesteps);

  clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_now);
  time_elapsed = timestamp_now.tv_sec - timestamp_start.tv_sec + 1E-9 * (timestamp_now.tv_nsec - timestamp_start.tv_nsec);
  printf("Computed %d time steps in %g s. Time steps per second: %g\n",
      num_timesteps, time_elapsed, num_timesteps/time_elapsed);
  printf("\n");

  process_results(states, num_states, num_nodes);
 
  free(states);
  free(parameters);

  return 0;
}