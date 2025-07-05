 #include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_listener.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <utility>

class KalmanFilter
{
public:
    KalmanFilter()
    {
        // Initialisierung der Matrizen und Vektoren
        A = Eigen::MatrixXd::Identity(5, 5); //  Identitätsmatrix
        B = Eigen::MatrixXd::Identity(5, 2); //  Identitätsmatrix
        C = Eigen::MatrixXd::Identity(5, 5); //  Identitätsmatrix
        Q = Eigen::MatrixXd::Identity(5, 5); //  Identitätsmatrix
        R = Eigen::MatrixXd::Identity(5, 5); //  Identitätsmatrix
        P = Eigen::MatrixXd::Identity(5, 5); //  Identitätsmatrix
        x = Eigen::VectorXd::Zero(5);        //  Nullvektor
    }

    void predict(const Eigen::VectorXd &u)
    {
        x = A * x + B * u;                     // Vorhersage des Zustands unter Verwendung der Steuereingabe (Odometrie)
        P = A * P * A.transpose() + Q;         // Vorhersage der Fehlerkovarianz
    }

    void update(const Eigen::VectorXd &measurement)
    {
        Eigen::MatrixXd K = P * C.transpose() * (C * P * C.transpose() + R).inverse(); // Kalman-Verstärkungsfaktor
        x = x + K * (measurement - C * x);                                             // Aktualisierung des Zustands
        P = (Eigen::MatrixXd::Identity(5, 5) - K * C) * P;                             // Aktualisierung der Fehlerkovarianz
    }

    Eigen::MatrixXd getCovariance() const
    {
        return P;
    }

private:
    Eigen::VectorXd x;       // Zustandsvektor
    Eigen::MatrixXd P;       // Kovarianzmatrix
    Eigen::MatrixXd A;       // Systemdynamik
    Eigen::MatrixXd B;       // Steuerungsmatrix
    Eigen::MatrixXd C;       // Beobachtungsmatrix
    Eigen::MatrixXd Q;       // Prozessrauschkovarianz
    Eigen::MatrixXd R;       // Messrauschkovarianz
};

int main(int argc, char** argv)
{
    // Setze die ROS-Log-Ebene auf WARN
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn))
    {
        ros::console::notifyLoggerLevelsChanged();
    }

    Eigen::MatrixXd Am;       // Systemdynamik
    Eigen::MatrixXd Bm;       // Steuerungsmatrix
    Eigen::MatrixXd Cm;       // Beobachtungsmatrix

    Am.setIdentity();
    Bm.setIdentity();
    Cm.setIdentity();


    // Zustandsvektor, Kovarianzmatrix und andere Variablen initialisieren
    Eigen::VectorXd xm(5);         // Zustandsvektor
    Eigen::MatrixXd Pm(5, 5);      // Kovarianzmatrix
    Eigen::VectorXd u(2);          // Steuerungsvektor
    Eigen::VectorXd z(5);          // Beobachtungsvektor

    xm.setZero();
    Pm.setIdentity();
    u.setZero();
    z.setZero();

    // Vektor für Zeiten und Positionen
    std::vector<std::pair<double, Eigen::VectorXd>> positions;

    // Initialisiere den ROS-Knoten
    ros::init(argc, argv, "turtlebot_control");
    ros::NodeHandle nh;

    // Erstelle einen Publisher für das Twist-Topic
    ros::Publisher cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 10);

    // Erstelle eine Instanz des Twist-Nachrichtentyps
    geometry_msgs::Twist cmd_vel_msg;

    // Setze die lineare Geschwindigkeit auf 0.2 (m/s)
    cmd_vel_msg.linear.x = 0.1;

    // Setze die Winkelgeschwindigkeit auf 0.0 (rad/s)
    cmd_vel_msg.angular.z = 0.0;

    // Sende den Befehl für 8 Sekunden
    ros::Time start_time = ros::Time::now();
    ros::Duration duration(8.0);

    Eigen::VectorXd previousPosition; // Variable für die vorherige Position

    while (ros::Time::now() - start_time < duration)
    {
        cmd_vel_pub.publish(cmd_vel_msg);
        ros::Duration(0.1).sleep(); // Pause für 0.1 Sekunden

        if (ros::Time::now() - start_time >= duration - ros::Duration(1.0))
        {
            // Setze die lineare und Winkelgeschwindigkeit auf 0, um den Roboter anzuhalten
            cmd_vel_msg.linear.x = 0.0;
            cmd_vel_msg.angular.z = 0.0;
        }

        // Aktuelle Position und Orientierung abrufen
        tf::TransformListener listener;
        tf::StampedTransform transform;
        try
        {
            // Warte auf die neueste Transformationsnachricht
            listener.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(1.0));
            listener.lookupTransform("map", "base_link", ros::Time(0), transform);

            // Extrahiere die Position (x, y) und die Orientierung (yaw)
            double x = transform.getOrigin().x();
            double y = transform.getOrigin().y();
            double yaw = tf::getYaw(transform.getRotation());

            ros::Time current_time = ros::Time::now();
            double elapsed_time = (current_time - start_time).toSec();

            std::cout << "Time: " << elapsed_time << ", Current Position (x, y): " << x << ", " << y << std::endl;
            std::cout << "Time: " << elapsed_time << ", Current Orientation (yaw): " << yaw << std::endl;

            // Anwendung des Kalman-Filters auf die Position
            if (!positions.empty())
            {
                // Die vorherige Position speichern
                previousPosition = positions.back().second;
                // Ausgabe der vorherigen Position
                 std::cout << "Vorherige Position: " << previousPosition << std::endl;
               

            }

            // Position und Zeit zu positions-Vektor hinzufügen
            positions.push_back(std::make_pair(elapsed_time, Eigen::VectorXd::Map(&x, 2)));
        }
        catch (tf::TransformException& ex)
        {
            ROS_ERROR("Fehler bei der Transformationsabfrage: %s", ex.what());
        }

        ros::spinOnce();
    }

    // // Überprüfe, ob mindestens eine Position vorhanden ist, bevor der Kalman-Filter angewendet wird
    // if (!positions.empty())
    // {
    //   // Die letzte Position im positions-Vektor abrufen
    //   Eigen::VectorXd currentPosition = positions.back().second;
       
    //   // Ausgabe der vorherigen Position
    //   std::cout << "Vorherige Position: " << previousPosition << std::endl;
       
    // }


    return 0;
}