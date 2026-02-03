#!/bin/bash

# Mount CIFS share if CIFS_PASSWORD is set (requires root)
if [ -n "$CIFS_PASSWORD" ]; then
    echo "Mounting CIFS share..."
    mount -t cifs //files.ugent.be/miepeete /mnt/share \
        -o username=miepeete,domain=UGENT,password="$CIFS_PASSWORD",vers=3.11,sec=ntlmssp,echo_interval=5,uid=dev,gid=dev
    if [ $? -eq 0 ]; then
        echo "CIFS share mounted successfully."
    else
        echo "Warning: Failed to mount CIFS share."
    fi
else
    echo "CIFS_PASSWORD not set, skipping CIFS mount."
fi

# Drop privileges and execute the main command as dev user
exec gosu dev "$@"
