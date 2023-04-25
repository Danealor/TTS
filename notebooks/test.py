import keyboard

input()
while True:
    event = keyboard.read_event()
    if event.event_type == 'up' and event.name == 'enter':
        break