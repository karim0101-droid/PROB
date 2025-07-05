#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h> // Hinzugefügt für die RPY-Konvertierung
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <random>
#include <vector>

// Struktur zur Darstellung eines einzelnen Partikels
struct Particle
{
    double x;
    double y;
    double theta;
    double weight;
};

class ParticleFilterNode
{
public:
    ParticleFilterNode(ros::NodeHandle &nh) : nh_(nh), map_received_(false)
    {
        // Parameter vom ROS Parameter Server laden
        loadParams();

        // Initialisierung der Partikel
        initializeParticles();

        // Publisher
        pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("/estimated_pose", 10);
        particles_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/particle_cloud", 10);

        // Subscriber
        map_sub_ = nh_.subscribe("/map", 1, &ParticleFilterNode::mapCallback, this);

        // Message Filter für synchronisierte Odometrie- und LaserScan-Daten
        odom_sub_.subscribe(nh_, odom_topic_, 10);
        scan_sub_.subscribe(nh_, scan_topic_, 10);

        sync_.reset(new message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::LaserScan>(odom_sub_, scan_sub_, 10));
        sync_->registerCallback(boost::bind(&ParticleFilterNode::sensorCallback, this, _1, _2));

        last_time_ = ros::Time::now();
    }

private:
    // --- ROS-spezifische Member-Variablen ---
    ros::NodeHandle nh_;
    ros::Publisher pose_pub_;
    ros::Publisher particles_pub_;
    ros::Subscriber map_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::LaserScan> scan_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::LaserScan>> sync_;
    ros::Time last_time_;
    nav_msgs::Odometry last_odom_;

    // --- Filter-Parameter ---
    int num_particles_;
    double initial_pose_x_;
    double initial_pose_y_;
    double initial_pose_theta_;
    double initial_cov_x_;
    double initial_cov_y_;
    double initial_cov_theta_;

    // Odometrie-Bewegungsmodell-Rauschparameter
    double alpha1_, alpha2_, alpha3_, alpha4_;

    // Likelihood-Field-Messmodell-Parameter
    double z_hit_, z_rand_;
    double sigma_hit_;

    // Topics
    std::string odom_topic_, scan_topic_;

    // --- Filter-Zustandsvariablen ---
    std::vector<Particle> particles_;
    bool map_received_;
    nav_msgs::OccupancyGrid map_;
    std::vector<double> likelihood_field_;

    // Zufallszahlengenerator
    std::default_random_engine generator_;

    /**
     * @brief Lädt Parameter vom ROS Parameter Server.
     */
    void loadParams()
    {
        nh_.param("num_particles", num_particles_, 500);
        nh_.param("initial_pose_x", initial_pose_x_, 0.0);
        nh_.param("initial_pose_y", initial_pose_y_, 0.0);
        nh_.param("initial_pose_theta", initial_pose_theta_, 0.0);
        nh_.param("initial_cov_x", initial_cov_x_, 0.25);
        nh_.param("initial_cov_y", initial_cov_y_, 0.25);
        nh_.param("initial_cov_theta", initial_cov_theta_, 0.1);

        nh_.param("odom_alpha1", alpha1_, 0.2);
        nh_.param("odom_alpha2", alpha2_, 0.2);
        nh_.param("odom_alpha3", alpha3_, 0.1);
        nh_.param("odom_alpha4", alpha4_, 0.1);

        nh_.param("laser_z_hit", z_hit_, 0.95);
        nh_.param("laser_z_rand", z_rand_, 0.05);
        nh_.param("laser_sigma_hit", sigma_hit_, 0.2);

        nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
        nh_.param<std::string>("scan_topic", scan_topic_, "/scan");
    }

    /**
     * @brief Initialisiert die Partikel um eine Startpose mit einer gegebenen Kovarianz.
     */
    void initializeParticles()
    {
        std::normal_distribution<double> dist_x(initial_pose_x_, initial_cov_x_);
        std::normal_distribution<double> dist_y(initial_pose_y_, initial_cov_y_);
        std::normal_distribution<double> dist_theta(initial_pose_theta_, initial_cov_theta_);

        particles_.resize(num_particles_);
        for (int i = 0; i < num_particles_; ++i)
        {
            particles_[i].x = dist_x(generator_);
            particles_[i].y = dist_y(generator_);
            particles_[i].theta = dist_theta(generator_);
            particles_[i].weight = 1.0 / num_particles_;
        }
        last_odom_.pose.pose.position.x = initial_pose_x_;
        last_odom_.pose.pose.position.y = initial_pose_y_;
        tf2::Quaternion q;
        q.setRPY(0, 0, initial_pose_theta_);
        last_odom_.pose.pose.orientation = tf2::toMsg(q);
    }

    /**
     * @brief Callback für die Karte. Wird nur einmal aufgerufen.
     * @param msg OccupancyGrid-Nachricht.
     */
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        if (map_received_)
            return;

        map_ = *msg;
        computeLikelihoodField();
        map_received_ = true;
        ROS_INFO("Map received and likelihood field computed.");
        map_sub_.shutdown(); // Subscriber wird nach dem ersten Empfang beendet
    }

    /**
     * @brief Berechnet das Likelihood-Field (Distanztransformation) für die Karte.
     * Dies wird nur einmal gemacht, um die Gewichtungsberechnung zu beschleunigen.
     */
    void computeLikelihoodField()
    {
        likelihood_field_.resize(map_.info.width * map_.info.height, -1.0);
        std::vector<std::pair<int, int>> obstacle_cells;

        // Finde alle Hinderniszellen
        for (unsigned int i = 0; i < map_.data.size(); ++i)
        {
            if (map_.data[i] > 50)
            { // Schwellenwert für "besetzt"
                int x = i % map_.info.width;
                int y = i / map_.info.width;
                obstacle_cells.push_back({x, y});
            }
        }

        // Berechne für jede Zelle die Distanz zum nächsten Hindernis
        for (unsigned int y = 0; y < map_.info.height; ++y)
        {
            for (unsigned int x = 0; x < map_.info.width; ++x)
            {
                double min_dist_sq = std::numeric_limits<double>::max();
                for (const auto &obs_cell : obstacle_cells)
                {
                    double dist_sq = std::pow(x - obs_cell.first, 2) + std::pow(y - obs_cell.second, 2);
                    if (dist_sq < min_dist_sq)
                    {
                        min_dist_sq = dist_sq;
                    }
                }
                likelihood_field_[y * map_.info.width + x] = std::sqrt(min_dist_sq) * map_.info.resolution;
            }
        }
    }

    /**
     * @brief Haupt-Callback, der durch synchronisierte Sensor-Nachrichten ausgelöst wird.
     * @param odom_msg Odometrie-Nachricht.
     * @param scan_msg LaserScan-Nachricht.
     */
    void sensorCallback(const nav_msgs::Odometry::ConstPtr &odom_msg, const sensor_msgs::LaserScan::ConstPtr &scan_msg)
    {
        if (!map_received_)
        {
            ROS_WARN("Waiting for map...");
            return;
        }

        predict(odom_msg);
        updateWeights(scan_msg);
        resample();

        publishPose();
        publishParticles();

        last_odom_ = *odom_msg;
    }

    /**
     * @brief Prediction-Schritt: Bewegt die Partikel basierend auf dem Odometrie-Bewegungsmodell.
     * @param odom_msg Die aktuelle Odometrie-Nachricht.
     */
    void predict(const nav_msgs::Odometry::ConstPtr &odom_msg)
    {
        double x_curr = odom_msg->pose.pose.position.x;
        double y_curr = odom_msg->pose.pose.position.y;

        tf2::Quaternion q_curr;
        tf2::fromMsg(odom_msg->pose.pose.orientation, q_curr);
        double roll_curr, pitch_curr, theta_curr;
        tf2::Matrix3x3(q_curr).getRPY(roll_curr, pitch_curr, theta_curr);

        double x_last = last_odom_.pose.pose.position.x;
        double y_last = last_odom_.pose.pose.position.y;

        tf2::Quaternion q_last;
        tf2::fromMsg(last_odom_.pose.pose.orientation, q_last);
        double roll_last, pitch_last, theta_last;
        tf2::Matrix3x3(q_last).getRPY(roll_last, pitch_last, theta_last);

        double delta_rot1 = atan2(y_curr - y_last, x_curr - x_last) - theta_last;
        double delta_trans = sqrt(pow(x_curr - x_last, 2) + pow(y_curr - y_last, 2));
        double delta_rot2 = theta_curr - theta_last - delta_rot1;

        for (auto &p : particles_)
        {
            double delta_rot1_noisy = delta_rot1 - sample(alpha1_ * pow(delta_rot1, 2) + alpha2_ * pow(delta_trans, 2));
            double delta_trans_noisy = delta_trans - sample(alpha3_ * pow(delta_trans, 2) + alpha4_ * pow(delta_rot1, 2) + alpha4_ * pow(delta_rot2, 2));
            double delta_rot2_noisy = delta_rot2 - sample(alpha1_ * pow(delta_rot2, 2) + alpha2_ * pow(delta_trans, 2));

            p.x += delta_trans_noisy * cos(p.theta + delta_rot1_noisy);
            p.y += delta_trans_noisy * sin(p.theta + delta_rot1_noisy);
            p.theta += delta_rot1_noisy + delta_rot2_noisy;
            p.theta = normalizeAngle(p.theta);
        }
    }

    /**
     * @brief Update-Schritt: Berechnet die Gewichte der Partikel neu basierend auf dem Laserscan.
     * @param scan_msg Die aktuelle LaserScan-Nachricht.
     */
    void updateWeights(const sensor_msgs::LaserScan::ConstPtr &scan_msg)
    {
        double total_weight = 0.0;

        for (auto &p : particles_)
        {
            p.weight = 1.0;
            // Wir verwenden nur jeden 5. Strahl, um die Berechnung zu beschleunigen
            for (size_t i = 0; i < scan_msg->ranges.size(); i += 5)
            {
                double range = scan_msg->ranges[i];
                if (range >= scan_msg->range_max || range <= scan_msg->range_min)
                {
                    continue;
                }

                double angle = scan_msg->angle_min + i * scan_msg->angle_increment;

                // Endpunkt des Laserstrahls in der Welt-Koordinaten
                double x_z = p.x + range * cos(p.theta + angle);
                double y_z = p.y + range * sin(p.theta + angle);

                // Konvertiere Welt-Koordinaten in Karten-Koordinaten (Pixel)
                int map_x = (x_z - map_.info.origin.position.x) / map_.info.resolution;
                int map_y = (y_z - map_.info.origin.position.y) / map_.info.resolution;

                if (map_x >= 0 && map_x < map_.info.width && map_y >= 0 && map_y < map_.info.height)
                {
                    double dist = likelihood_field_[map_y * map_.info.width + map_x];

                    // Wahrscheinlichkeit basierend auf der Distanz zum nächsten Hindernis
                    double prob = gaussianPdf(dist, 0, sigma_hit_);
                    p.weight *= (z_hit_ * prob + z_rand_ / scan_msg->range_max);
                }
                else
                {
                    p.weight *= (z_rand_ / scan_msg->range_max);
                }
            }
            total_weight += p.weight;
        }

        // Normalisiere die Gewichte
        if (total_weight > 0.0)
        {
            for (auto &p : particles_)
            {
                p.weight /= total_weight;
            }
        }
        else
        {
            // Wenn alle Gewichte 0 sind, Gewichte zurücksetzen
            for (auto &p : particles_)
            {
                p.weight = 1.0 / num_particles_;
            }
        }
    }

    /**
     * @brief Resampling-Schritt: Erzeugt eine neue Partikelmenge durch Ziehen aus der alten Menge
     * basierend auf den Gewichten. Verwendet den Low-Variance-Sampler.
     */
    void resample()
    {
        std::vector<Particle> new_particles;
        new_particles.reserve(num_particles_);

        double r = (double)rand() / RAND_MAX / num_particles_;
        double c = particles_[0].weight;
        int i = 0;

        for (int m = 0; m < num_particles_; ++m)
        {
            double u = r + (double)m / num_particles_;
            while (u > c)
            {
                i++;
                c += particles_[i].weight;
            }
            new_particles.push_back(particles_[i]);
        }
        particles_ = new_particles;

        // Nach dem Resampling haben alle Partikel das gleiche Gewicht
        for (auto &p : particles_)
        {
            p.weight = 1.0 / num_particles_;
        }
    }

    /**
     * @brief Schätzt die Pose (Mittelwert und Kovarianz) aus der Partikelwolke.
     */
    void publishPose()
    {
        geometry_msgs::PoseWithCovarianceStamped estimated_pose;
        estimated_pose.header.stamp = ros::Time::now();
        estimated_pose.header.frame_id = "map";

        double mean_x = 0.0, mean_y = 0.0;
        double mean_cos_theta = 0.0, mean_sin_theta = 0.0;

        for (const auto &p : particles_)
        {
            mean_x += p.x;
            mean_y += p.y;
            mean_cos_theta += cos(p.theta);
            mean_sin_theta += sin(p.theta);
        }
        mean_x /= num_particles_;
        mean_y /= num_particles_;
        mean_cos_theta /= num_particles_;
        mean_sin_theta /= num_particles_;
        double mean_theta = atan2(mean_sin_theta, mean_cos_theta);

        estimated_pose.pose.pose.position.x = mean_x;
        estimated_pose.pose.pose.position.y = mean_y;
        tf2::Quaternion q;
        q.setRPY(0, 0, mean_theta);
        estimated_pose.pose.pose.orientation = tf2::toMsg(q);

        // Kovarianz berechnen
        double cov_xx = 0.0, cov_yy = 0.0, cov_tt = 0.0;
        double cov_xy = 0.0, cov_xt = 0.0, cov_yt = 0.0;

        for (const auto &p : particles_)
        {
            cov_xx += pow(p.x - mean_x, 2);
            cov_yy += pow(p.y - mean_y, 2);
            double d_theta = normalizeAngle(p.theta - mean_theta);
            cov_tt += pow(d_theta, 2);
            cov_xy += (p.x - mean_x) * (p.y - mean_y);
            cov_xt += (p.x - mean_x) * d_theta;
            cov_yt += (p.y - mean_y) * d_theta;
        }
        cov_xx /= num_particles_;
        cov_yy /= num_particles_;
        cov_tt /= num_particles_;
        cov_xy /= num_particles_;
        cov_xt /= num_particles_;
        cov_yt /= num_particles_;

        estimated_pose.pose.covariance[0] = cov_xx;
        estimated_pose.pose.covariance[1] = cov_xy;
        estimated_pose.pose.covariance[5] = cov_xt;
        estimated_pose.pose.covariance[6] = cov_xy;
        estimated_pose.pose.covariance[7] = cov_yy;
        estimated_pose.pose.covariance[11] = cov_yt;
        estimated_pose.pose.covariance[30] = cov_xt;
        estimated_pose.pose.covariance[31] = cov_yt;
        estimated_pose.pose.covariance[35] = cov_tt;

        pose_pub_.publish(estimated_pose);
    }

    /**
     * @brief Veröffentlicht die Partikelwolke zur Visualisierung in RViz.
     */
    void publishParticles()
    {
        geometry_msgs::PoseArray pose_array;
        pose_array.header.stamp = ros::Time::now();
        pose_array.header.frame_id = "map";
        pose_array.poses.resize(num_particles_);

        for (int i = 0; i < num_particles_; ++i)
        {
            pose_array.poses[i].position.x = particles_[i].x;
            pose_array.poses[i].position.y = particles_[i].y;
            tf2::Quaternion q;
            q.setRPY(0, 0, particles_[i].theta);
            pose_array.poses[i].orientation = tf2::toMsg(q);
        }
        particles_pub_.publish(pose_array);
    }

    // --- Hilfsfunktionen ---

    /**
     * @brief Normalisiert einen Winkel auf den Bereich [-PI, PI].
     * @param angle Der zu normalisierende Winkel.
     * @return Der normalisierte Winkel.
     */
    double normalizeAngle(double angle)
    {
        while (angle > M_PI)
            angle -= 2.0 * M_PI;
        while (angle < -M_PI)
            angle += 2.0 * M_PI;
        return angle;
    }

    /**
     * @brief Erzeugt eine Zufallszahl aus einer Normalverteilung mit Varianz b.
     * @param b Die Varianz der Verteilung.
     * @return Eine Zufallszahl.
     */
    double sample(double b)
    {
        std::normal_distribution<double> dist(0.0, sqrt(b));
        return dist(generator_);
    }

    /**
     * @brief Berechnet den Wert der Gaußschen Wahrscheinlichkeitsdichtefunktion.
     * @param x Der Wert, für den die Dichte berechnet werden soll.
     * @param mu Der Mittelwert der Verteilung.
     * @param sigma Die Standardabweichung der Verteilung.
     * @return Der Wert der Dichtefunktion.
     */
    double gaussianPdf(double x, double mu, double sigma)
    {
        return (1.0 / (sigma * sqrt(2.0 * M_PI))) * exp(-0.5 * pow((x - mu) / sigma, 2));
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pf_node"); // Geänderter Node-Name
    ros::NodeHandle nh("~");          // Privater Node Handle, um Parameter zu lesen
    ParticleFilterNode node(nh);
    ros::spin();

    return 0;
}