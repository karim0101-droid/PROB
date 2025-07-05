#!/usr/bin/env bash
# ----------------[ USER-PARAMETER ]-----------------
RECORD_START_DELAY_SEC=5 # Verzögerung bis rosbag nach dem Launch beginnt
RECORD_DURATION_SEC=85   # Dauer der Rosbag-Aufnahme in Sekunden. Anpassen!
                         # Sollte lang genug sein, damit Roboter alle Ziele abfährt.

# Basisname für die Bag-Datei und Plots
BASE_NAME="run_$(date +%Y%m%d_%H%M%S)"

# Verzeichnis für die Bag-Datei und Plots
BAG_AND_PLOT_DIR="$(dirname "$(dirname "$0")")/plots"

# Topics, die aufgezeichnet werden sollen (als Bash-Array)
declare -a TOPICS_TO_RECORD=(
    "/odom"
    "/kf_prediction"
    "/ekf_prediction"
    "/pf_prediction"
    "/kf_velocity_prediction"
    "/ekf_velocity_prediction"
    "/pf_velocity_prediction"
)

# Pfad zum Python-Plotting-Skript
PLOT_SCRIPT_PATH="$(dirname "$0")/plot_trajectories.py"

# ---------------------------------------------------

mkdir -p "$BAG_AND_PLOT_DIR" # Zielordner für Bags und PNGs erstellen

echo ">>> Skript 'record_and_plot.sh' gestartet."
echo ">>> Warte ${RECORD_START_DELAY_SEC} Sekunden, bevor rosbag beginnt..."
sleep ${RECORD_START_DELAY_SEC}

BAG_PID=""
ACTUAL_BAG_FILE="" # Variable, um den tatsächlich erstellten Dateinamen zu speichern

# TRAF-Handler: Wird bei SIGINT (Ctrl+C), SIGTERM (kill) aufgerufen
function cleanup {
  echo ""
  echo ">>> cleanup-Funktion aufgerufen."
  
  if [ -n "$BAG_PID" ] && ps -p "$BAG_PID" > /dev/null; then
    echo ">>> rosbag-Prozess (PID: $BAG_PID) wird beendet..."
    # Sende SIGINT an rosbag, damit es eine saubere Beendigung versucht
    kill -s SIGINT "$BAG_PID"
    wait "$BAG_PID" 2>/dev/null
    echo ">>> rosbag-Prozess beendet."
  else
    echo ">>> rosbag-Prozess nicht gefunden oder bereits beendet."
  fi

  # Wenn der ACTUAL_BAG_FILE noch nicht gesetzt ist, versuchen wir ihn hier zu finden
  if [ -z "$ACTUAL_BAG_FILE" ]; then
    # Suche die neueste .bag-Datei im Verzeichnis, die mit unserem BASE_NAME beginnt
    ACTUAL_BAG_FILE=$(find "$BAG_AND_PLOT_DIR" -maxdepth 1 -name "${BASE_NAME}*.bag" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
  fi

  # Überprüfe, ob die Bag-Datei existiert und nicht leer ist
  if [ -s "$ACTUAL_BAG_FILE" ]; then # -s prüft, ob die Datei existiert und nicht leer ist
    echo ">>> Rosbag-Aufnahme beendet (manuell abgebrochen oder Timeout). Generiere Plots..."

    PLOT_OUTPUT_PREFIX="${BAG_AND_PLOT_DIR}/${BASE_NAME}" # Prefix bleibt unser ursprünglicher BASE_NAME

    python3 "${PLOT_SCRIPT_PATH}" "${ACTUAL_BAG_FILE}" \
           --gt_pose  "${TOPICS_TO_RECORD[0]}" \
           --kf_pose  "${TOPICS_TO_RECORD[1]}" \
           --ekf_pose "${TOPICS_TO_RECORD[2]}" \
           --pf_pose  "${TOPICS_TO_RECORD[3]}" \
           --gt_vel   "${TOPICS_TO_RECORD[0]}" \
           --kf_vel   "${TOPICS_TO_RECORD[4]}" \
           --ekf_vel  "${TOPICS_TO_RECORD[5]}" \
           --pf_vel   "${TOPICS_TO_RECORD[6]}" \
           --out_prefix "${PLOT_OUTPUT_PREFIX}"
    
    echo ">>> Plots gespeichert unter ${BAG_AND_PLOT_DIR}/${BASE_NAME}_*.png"
  else
    echo "FEHLER: Bag-Datei '$ACTUAL_BAG_FILE' existiert nicht oder ist leer. Plots können nicht generiert werden."
  fi

  echo ">>> Prozess abgeschlossen (manuell abgebrochen oder Timeout)."
  exit 0
}

# Registriere die Cleanup-Funktion für Signale
trap cleanup SIGINT SIGTERM

echo ">>> Starte rosbag-Aufnahme mit Präfix '${BAG_AND_PLOT_DIR}/${BASE_NAME}' für ${RECORD_DURATION_SEC}s..."
# Führe rosbag im Hintergrund aus und leite seine Ausgabe an eine temporäre Datei um
# Damit wir den tatsächlich generierten Dateinamen aus dem Log abfangen können.
# --output-prefix (-O) sorgt dafür, dass rosbag einen Zeitstempel anhängt.
( rosbag record -O "${BAG_AND_PLOT_DIR}/${BASE_NAME}" --duration=${RECORD_DURATION_SEC} \
    "${TOPICS_TO_RECORD[@]}" 2>&1 | tee /tmp/rosbag_output_${BASE_NAME}.log ) & # 2>&1 | tee leitet stdout und stderr an Logfile und Bildschirm
BAG_PID=$!

echo ">>> Rosbag läuft im Hintergrund (PID: $BAG_PID). Warte auf den tatsächlichen Dateinamen..."

# Warte kurz, bis rosbag den Dateinamen in sein Log geschrieben hat
sleep 2

# Den tatsächlichen Dateinamen aus dem rosbag-Log extrahieren
# Suche nach der Zeile "Recording to '<path/filename.bag>'.", extrahiere den Pfad
ACTUAL_BAG_FILE=$(grep "Recording to '" "/tmp/rosbag_output_${BASE_NAME}.log" | head -n 1 | sed -e "s/.*Recording to '\([^']*\)'.*/\1/")

if [ -z "$ACTUAL_BAG_FILE" ]; then
  echo "WARNUNG: Konnte den tatsächlichen Namen der Bag-Datei nicht aus dem rosbag-Log ermitteln."
  echo "Versuche stattdessen, die neueste Bag-Datei im Ordner '$BAG_AND_PLOT_DIR' zu finden..."
  ACTUAL_BAG_FILE=$(find "$BAG_AND_PLOT_DIR" -maxdepth 1 -name "${BASE_NAME}*.bag" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -z "$ACTUAL_BAG_FILE" ]; then
    echo "FEHLER: Konnte keine Bag-Datei finden, die mit '${BASE_NAME}' beginnt. Plots werden nicht generiert."
    # Exit here, da keine Bag-Datei gefunden wurde
    kill -s SIGINT "$BAG_PID" 2>/dev/null # rosbag beenden, wenn es noch läuft
    exit 1
  fi
fi

echo ">>> Bag-Datei wird aufgezeichnet nach: ${ACTUAL_BAG_FILE}"
echo ">>> Drücke Ctrl+C, um die Aufnahme zu beenden und Plots zu generieren (oder warte ${RECORD_DURATION_SEC}s für automatisches Ende)."

wait $BAG_PID # Warte, bis rosbag beendet ist (durch --duration oder manuell durch Ctrl+C)

# Wenn hierher gelangt, ist rosbag beendet. Plotting wird vom Trap-Handler (cleanup)
# oder im nachfolgenden Fall (rosbag hat sich sauber beendet) übernommen.
# Der Trap-Handler wird auch aufgerufen, wenn Ctrl+C gedrückt wird, was das Skript beendet.
# Wenn rosbag durch --duration selbst beendet wird (Exit-Code 0),
# dann wird der Code HIER nach dem `wait` ausgeführt.

# Entferne den Trap, damit weitere Signale das Skript nicht erneut verarbeiten
trap - SIGINT SIGTERM

echo ">>> Rosbag-Aufnahme beendet (durch --duration oder extern). Generiere Plots (falls nicht schon in cleanup erfolgt)..."

# Diese `if`-Abfrage ist der Fall, wenn rosbag sauber durch --duration beendet wurde
if [ $? -eq 0 ]; then # Rosbag wurde sauber beendet (Exit-Code 0)
    # Wenn Bag-Datei leer ist (passiert manchmal bei sehr kurzen Aufnahmen), lösche sie und melde Fehler
    if [ ! -s "$ACTUAL_BAG_FILE" ]; then
        echo "FEHLER: Bag-Datei '$ACTUAL_BAG_FILE' ist leer. Lösche leere Datei. Plots können nicht generiert werden."
        rm -f "$ACTUAL_BAG_FILE"
        exit 1
    fi

    echo ">>> Generiere Plots..."
    PLOT_OUTPUT_PREFIX="${BAG_AND_PLOT_DIR}/${BASE_NAME}"
    python3 "${PLOT_SCRIPT_PATH}" "${ACTUAL_BAG_FILE}" \
           --gt_pose  "${TOPICS_TO_RECORD[0]}" \
           --kf_pose  "${TOPICS_TO_RECORD[1]}" \
           --ekf_pose "${TOPICS_TO_RECORD[2]}" \
           --pf_pose  "${TOPICS_TO_RECORD[3]}" \
           --gt_vel   "${TOPICS_TO_RECORD[0]}" \
           --kf_vel   "${TOPICS_TO_RECORD[4]}" \
           --ekf_vel  "${TOPICS_TO_RECORD[5]}" \
           --pf_vel   "${TOPICS_TO_RECORD[6]}" \
           --out_prefix "${PLOT_OUTPUT_PREFIX}"
    echo ">>> Plots gespeichert unter ${BAG_AND_PLOT_DIR}/${BASE_NAME}_*.png"
else
    echo ">>> Rosbag wurde nicht sauber beendet oder timeout (Plots wurden eventuell schon in cleanup versucht)."
fi

echo ">>> Prozess abgeschlossen."