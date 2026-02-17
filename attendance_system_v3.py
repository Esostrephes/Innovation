import configparser

# Configuration management using configparser
class ConfigManager:
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get(self, section, option):
        return self.config.get(section, option)


class AttendanceSystem:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def mark_attendance(self, student_id):
        # Example method to mark attendance
        attendance_record = {'id': student_id, 'status': 'present'}
        print(f'Marking attendance for {student_id}: {attendance_record}')

    def get_attendance(self):
        # Example method to get attendance
        print('Getting attendance records...')


if __name__ == '__main__':
    config_manager = ConfigManager()
    attendance_system = AttendanceSystem(config_manager)
    attendance_system.mark_attendance('12345')
    attendance_system.get_attendance()