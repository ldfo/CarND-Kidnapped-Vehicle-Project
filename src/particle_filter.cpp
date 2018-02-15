/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static void get_pred_landmarks(std::vector<LandmarkObs> &pred_landmarks, const Map &map_landmarks);

static void to_map_coordinates(std::vector<LandmarkObs> &meas_landmarks,
        const std::vector<LandmarkObs> &observations,
        Particle particle);

static double calc_weight(const std::vector<LandmarkObs> &pred_landmarks,
        const std::vector<LandmarkObs> &meas_landmarks,
        double std_x,
        double std_y);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // initialize ParticleFilter
    num_particles = 50;
    particles.resize(num_particles);
    weights.resize(num_particles);

    //init particles with normal distribution
    double init_weight = 1.0;

    normal_distribution<double> dist_x(x, std[0]), dist_y(y, std[1]), dist_theta(theta, std[2]);
    default_random_engine gen;
    for(int i=0; i< num_particles; i++){
        particles[i] = {i, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight};
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // using the CTRV model
    // predict state using time, gps uncertainty, vect1 and vect2
    // 1. Process next state according to CTRV-model
    for (int i = 0; i < num_particles; i++) {
        // in case of yaw_rate == 0
        if (fabs(yaw_rate) < 0.001) {
            particles[i].x += velocity * cos(particles[i].theta) * delta_t;
            particles[i].y += velocity * sin(particles[i].theta) * delta_t;
        }
        // in case of yaw_rate != 0
        else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta));
        }
        particles[i].theta = particles[i].theta + yaw_rate * delta_t;
    }
    // 2. Add gaussian noise
    normal_distribution<double> dist_x(0, std_pos[0]), dist_y(0, std_pos[1]), dist_theta(0, std_pos[2]);
    
    default_random_engine gen;

    for (int i = 0; i < num_particles; i++) {
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); i++)
    {
        double min_dist = 40000;
        int min_index = -1;
        for (int j = 0; j < predicted.size(); j++)
        {
            // calculate dist
            double dist = sqrt(pow(observations[i].x - predicted[j].x, 2) +
                               pow(observations[i].y - predicted[j].y, 2));

            if (min_dist > dist)
            {
                min_dist = dist;
                min_index = j;
            }
        }

            observations[i].id = predicted[min_index].id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    // get predicted landmarks
    std::vector<LandmarkObs> pred_landmarks(num_particles);
    get_pred_landmarks(pred_landmarks, map_landmarks);
    for (int i = 0; i < num_particles; i++)
    {
        // change coords
        std::vector<LandmarkObs> meas_landmarks(observations.size());
        to_map_coordinates(meas_landmarks, observations, particles[i]);
        // nearest landmarks
        dataAssociation(pred_landmarks, meas_landmarks);
        // update weights
        particles[i].weight = calc_weight(pred_landmarks, meas_landmarks, std_landmark[0], std_landmark[1]);
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // resample particles 
    default_random_engine gen;
    vector<Particle> resampled_particles(num_particles);
    vector<double> weights(num_particles);

    // Get particles weights
    for (int i = 0; i < num_particles; i++) {
        weights[i] = particles[i].weight;
    }


    discrete_distribution<int> index(weights.begin(), weights.end());
    for (int j=0; j<num_particles;j++){
        int i = index(gen);
        resampled_particles[j] = particles[i];
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle associations
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> vect;
    vect = best.associations;
    stringstream ss;
    copy( vect.begin(), vect.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
    vector<double> vect;
    vect = best.sense_x;
    stringstream ss;
    copy( vect.begin(), vect.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> vect;
    vect = best.sense_y;
    stringstream ss;
    copy( vect.begin(), vect.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


static void get_pred_landmarks(std::vector<LandmarkObs> &pred_landmarks, const Map &map_landmarks)
{
     // Get pred_landmarks
    for (int i = 0; i < map_landmarks.landmark_list.size(); i++){
        LandmarkObs obs;
        // id
        obs.id = map_landmarks.landmark_list[i].id_i;
        // x
        obs.x = map_landmarks.landmark_list[i].x_f;
        // y
        obs.y = map_landmarks.landmark_list[i].y_f;
        pred_landmarks[i] = obs;
    }
}

static void to_map_coordinates(std::vector<LandmarkObs> &meas_landmarks,
        const std::vector<LandmarkObs> &observations, Particle particle){
     // sensor input from particle to map coordinates
    double xp;
    double yp;
    double theta_p;
    double xc;
    double yc;
    double xm;
    double ym;

    xp = particle.x;
    yp = particle.y;
    theta_p = particle.theta;

    for (int i = 0; i < observations.size(); i++)
    {
        xc = observations[i].x;
        yc = observations[i].y;

        xm = xp + cos(theta_p)*xc - sin(theta_p)*yc;
        ym = yp + sin(theta_p)*xc + cos(theta_p)*yc;

        LandmarkObs obs;
        obs.x = xm;
        obs.y = ym;

        obs.id = observations[i].id;
        meas_landmarks[i] = obs;
    }
}

static double calc_weight(const std::vector<LandmarkObs> &pred_landmarks,
        const std::vector<LandmarkObs> &meas_landmarks, double std_x, double std_y){
    // calculate weight
    double weight;
    double na;
    double nb;
    double gauss_norm;
    double pr_x, pr_y;
    double o_x;
    double o_y;
    double obs_w;

    weight = 1.0;
    // tried with several coeffs 2.5 gave the best result for me
    na = 2.5 * std_x * std_x;
    nb = 2.5 * std_y * std_y;
    gauss_norm = 2.5 * M_PI * std_x * std_y;

    for (int i=0; i < meas_landmarks.size(); i++){
        o_x = meas_landmarks[i].x;
        o_y = meas_landmarks[i].y;

        for (int i2 = 0; i2 < pred_landmarks.size(); i2++) {

            if (pred_landmarks[i2].id == meas_landmarks[i].id) {
                pr_x = pred_landmarks[i2].x;
                pr_y = pred_landmarks[i2].y;
                break;
            }
        }
        obs_w =  pow(pr_x-o_x,2)/na + (pow(pr_y-o_y,2)/nb);
        obs_w = 1/gauss_norm * exp( -obs_w );
        weight *= obs_w;
    }
    return weight;
}
