"""Close the old five-port pre11 state without changing any existing port."""
def closed_ports(raw,ports):
    assert len(ports)==5
    return list(ports)+[raw.double()-sum(ports)]
